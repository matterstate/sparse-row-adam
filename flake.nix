{
  description = "Bounded Rust CPU updates for sparse bf16 memory tables";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];
      systems = [ "x86_64-linux" ];

      perSystem = { pkgs, ... }: {
        packages.default =
          pkgs.runCommand "sparse-row-adam"
            {
              nativeBuildInputs = [
                pkgs.rustc
                pkgs.stdenv.cc
              ];
            }
            ''
              mkdir -p $out/lib $out/include
              rustc --edition=2021 --crate-type cdylib -C opt-level=3 -C codegen-units=1 \
                ${./src/lib.rs} -o $out/lib/libsparse_row_adam.so
              cp ${./include/sparse_row_adam.h} $out/include/sparse_row_adam.h
            '';

        formatter = pkgs.nixfmt;

        devshells.default = {
          name = "sparse-row-adam";
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
            pkgs.clippy
            pkgs.stdenv.cc
            pkgs.clang-tools
            pkgs.nixfmt
            pkgs.taplo
            pkgs.jq
            (pkgs.python313.withPackages (python: [
              python.numpy
              python.pytest
              python.ruff
              python.mdformat
              python.mdformat-gfm
              python.mdformat-footnote
              python.mdformat-frontmatter
            ]))
          ];
        };
      };
    };
}
