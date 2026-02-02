{
  description = "My rust assistant";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    dream2nix = {
      url = "github:nix-community/dream2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:oxalica/rust-overlay";
    };
  };

  outputs =
    inputs@{
      self,
      devshell,
      dream2nix,
      nixpkgs,
      flake-parts,
      treefmt-nix,
      rust-overlay,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      imports = [
        devshell.flakeModule
        treefmt-nix.flakeModule
      ];
      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              (import inputs.rust-overlay)
            ];
            config.allowUnfree = true;
          };
          treefmt.programs = {
            nixfmt.enable = true;
            rustfmt.enable = true;
          };
          # packages.default = {};
          devshells.default = {
            name = "myDevShell";
            devshell.startup = {
              dream2nixEnv.text = ''
                export NIX_PATH=nixpkgs=${inputs.nixpkgs}
              '';
            };
            packages = with pkgs; [
              gitleaks
              rust-analyzer
              rust-bin.nightly.latest.default
            ];
          };
        };

    };
}
