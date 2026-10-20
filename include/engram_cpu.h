#ifndef ENGRAM_CPU_H
#define ENGRAM_CPU_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t rows;
  size_t lookups;
  double learning_rate;
  double weight_decay;
} EngramStepOptions;

/* Arrays must be aligned, disjoint and valid for the duration of the call.
   bits: rows * 64 writable uint16 bf16 bit patterns.
   moments: rows writable float32 second moments.
   indices: lookups int64 row indices.
   gradients: lookups * 64 float32 values in occurrence order.
   Returns touched rows, or -1 dimensions/rates, -2 index, -3 state, -4
   workspace. Checked errors do not mutate bits or moments. Concurrent calls
   must not share writable arrays. The caller owns pointer validity and
   allocation lengths. */
int64_t engram_step(uint16_t *bits, float *moments, const int64_t *indices,
                    const float *gradients, const EngramStepOptions *options);
#endif
