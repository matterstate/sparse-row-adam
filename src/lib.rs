use std::slice;

const WIDTH: usize = 64;
const ROWS_MAX: usize = 128_000_000;
const LOOKUPS_MAX: usize = 1_048_576;
const BFLOAT16_OVERFLOW_BITS: u32 = 0x7f7f_8000;
const EMPTY: u32 = u32::MAX;

#[repr(C)]
pub struct StepOptions {
    rows: usize,
    lookups: usize,
    learning_rate: f64,
    weight_decay: f64,
}

/// # Safety
/// Pointers must name disjoint, aligned arrays of the declared lengths for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn sparse_row_adam_step(
    bits: *mut u16,
    moments: *mut f32,
    indices: *const i64,
    gradients: *const f32,
    options: *const StepOptions,
) -> i64 {
    if options.is_null() {
        return -1;
    }
    let options = unsafe { &*options };
    let rows = options.rows;
    let lookups = options.lookups;
    let learning_rate = options.learning_rate;
    let weight_decay = options.weight_decay;
    if rows == 0 || rows > ROWS_MAX || lookups > LOOKUPS_MAX {
        return -1;
    }
    if !learning_rate.is_finite() || !(0.0..=1.0).contains(&learning_rate) || learning_rate == 0.0 {
        return -1;
    }
    if !weight_decay.is_finite() || !(0.0..=1.0).contains(&weight_decay) {
        return -1;
    }
    if lookups == 0 {
        return 0;
    }
    if bits.is_null() || moments.is_null() || indices.is_null() || gradients.is_null() {
        return -1;
    }
    let indices = unsafe { slice::from_raw_parts(indices, lookups) };
    let gradients = unsafe { slice::from_raw_parts(gradients, lookups * WIDTH) };
    if indices.iter().any(|&row| row < 0 || row as usize >= rows) {
        return -2;
    }
    let bits = unsafe { slice::from_raw_parts_mut(bits, rows * WIDTH) };
    let moments = unsafe { slice::from_raw_parts_mut(moments, rows) };
    sparse_step(
        bits,
        moments,
        indices,
        gradients,
        learning_rate,
        weight_decay,
    )
}

fn sparse_step(
    bits: &mut [u16],
    moments: &mut [f32],
    indices: &[i64],
    gradients: &[f32],
    learning_rate: f64,
    weight_decay: f64,
) -> i64 {
    assert_eq!(bits.len(), moments.len() * WIDTH);
    assert_eq!(gradients.len(), indices.len() * WIDTH);
    let capacity = (2 * indices.len()).next_power_of_two();
    let mut slots = Vec::new();
    let mut rows = Vec::new();
    let mut sums: Vec<[f32; WIDTH]> = Vec::new();
    let mut next_moments = Vec::new();
    if slots.try_reserve_exact(capacity).is_err()
        || rows.try_reserve_exact(indices.len()).is_err()
        || sums.try_reserve_exact(indices.len()).is_err()
        || next_moments.try_reserve_exact(indices.len()).is_err()
    {
        return -4;
    }
    slots.resize(capacity, EMPTY);
    let shift = 64 - capacity.trailing_zeros();
    for (position, &row) in indices.iter().enumerate() {
        let mut bucket = ((row as u64).wrapping_mul(0x9e3779b97f4a7c15) >> shift) as usize;
        let mut found = None;
        for _ in 0..capacity {
            let entry = slots[bucket];
            if entry == EMPTY {
                let entry = rows.len();
                slots[bucket] = entry as u32;
                rows.push(row as usize);
                sums.push([0.0; WIDTH]);
                found = Some(entry);
                break;
            }
            if rows[entry as usize] == row as usize {
                found = Some(entry as usize);
                break;
            }
            bucket = (bucket + 1) & (capacity - 1);
        }
        let Some(entry) = found else { return -4 };
        let gradient = &gradients[position * WIDTH..(position + 1) * WIDTH];
        for (sum, &value) in sums[entry].iter_mut().zip(gradient) {
            *sum += value;
        }
    }
    let rate = learning_rate as f32;
    let decay = (learning_rate * weight_decay) as f32;
    for (&row, sum) in rows.iter().zip(&mut sums) {
        let mut partial = [0.0f32; 8];
        for (index, value) in partial.iter_mut().enumerate() {
            *value = sum[index] * sum[index];
        }
        for chunk in sum[8..].as_chunks::<8>().0 {
            for (partial, &value) in partial.iter_mut().zip(chunk) {
                *partial += value * value;
            }
        }
        let mean = (((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7])))
            / WIDTH as f32;
        let moment = moments[row] * 0.95f32 + 0.05f32 * mean;
        if !moments[row].is_finite() || moments[row] < 0.0 || !moment.is_finite() {
            return -3;
        }
        next_moments.push(moment);
        let denominator = moment.sqrt() + 1e-8f32;
        for (column, value) in sum.iter_mut().enumerate() {
            let weight = f32::from_bits((bits[row * WIDTH + column] as u32) << 16);
            let mut update = rate * *value / denominator;
            if update * weight > 0.0 {
                update += weight * decay;
            }
            *value = weight - update;
        }
        let magnitude_max = sum.iter().fold(0, |magnitude, value| {
            magnitude.max(value.to_bits() & 0x7fff_ffff)
        });
        if magnitude_max >= BFLOAT16_OVERFLOW_BITS {
            return -3;
        }
    }
    for ((&row, values), &moment) in rows.iter().zip(&sums).zip(&next_moments) {
        moments[row] = moment;
        for (column, &value) in values.iter().enumerate() {
            bits[row * WIDTH + column] = rounded_bits(value);
        }
    }
    rows.len() as i64
}

fn rounded_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1)) >> 16) as u16
}
