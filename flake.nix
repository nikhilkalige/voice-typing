{
  description = "Push-to-talk voice typing via parakeet.cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # required for CUDA packages
      };

      inherit (pkgs) lib;

      python = pkgs.python312;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      pythonSet =
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (lib.composeManyExtensions [
            pyproject-build-systems.overlays.default  # hatchling, wheel, setuptools, …
            overlay                                   # packages from uv.lock
          ]);

      venv = pythonSet.mkVirtualEnv "voice-typing-env" workspace.deps.default;

      parakeet-cpp = pkgs.callPackage ./nix/parakeet-cpp.nix {
        inherit (pkgs) cudaPackages;
      };

      runtimeLibPath = pkgs.lib.makeLibraryPath [
        pkgs.cudaPackages.cuda_cudart
        pkgs.cudaPackages.libcublas
        pkgs.libpulseaudio   # soundcard uses CFFI dlopen("libpulse.so") at runtime
      ];
    in
    {
      packages.${system} = {
        inherit parakeet-cpp;

        default = pkgs.writeShellApplication {
          name = "voicetype";
          runtimeInputs = [ pkgs.libnotify ];
          text = ''
            export VOICETYPE_PARAKEET_LIB="${parakeet-cpp}/lib/libparakeet.so"
            export LD_LIBRARY_PATH="${runtimeLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec "${venv}/bin/voicetype" "$@"
          '';
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.uv ];
      };
    };
}
