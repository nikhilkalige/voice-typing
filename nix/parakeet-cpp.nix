{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  patchelf,
  cudaPackages,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "parakeet-cpp";
  version = "0.3.2";

  src = fetchFromGitHub {
    owner = "mudler";
    repo = "parakeet.cpp";
    rev = "v${finalAttrs.version}";
    hash = "sha256-bXQl3zOuTnjxmfJcYJarpns5ehldUmGSms+AiMh/TUM=";
    fetchSubmodules = true;
  };

  # The original script uses `git apply` and checks for a .git directory, both
  # of which are absent in the Nix sandbox. Replace with a patch-based version.
  postPatch = ''
    install -m755 ${./apply_ggml_patches.sh} scripts/apply_ggml_patches.sh
  '';

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
    patchelf
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    stdenv.cc.cc.lib  # libgomp.so.1 (OpenMP runtime, used by ggml)
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DPARAKEET_SHARED=ON"
    "-DPARAKEET_BUILD_CLI=OFF"
    "-DPARAKEET_BUILD_TESTS=OFF"
    "-DPARAKEET_BUILD_SERVER=OFF"
    "-DPARAKEET_GGML_CUDA=ON"
  ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib $out/include

    # Copy libparakeet.so and all ggml shared libs it links against.
    find . -name 'lib*.so*' -not -type d | while read -r f; do
      cp -P "$f" $out/lib/
    done

    # Rewrite RPATHs from the /build/ sandbox paths to Nix store paths.
    # libcuda.so (the GPU driver) is not in the Nix store; it must come
    # from the system via LD_LIBRARY_PATH at runtime.
    rpath="$out/lib:${cudaPackages.cuda_cudart}/lib:${cudaPackages.libcublas}/lib:${stdenv.cc.cc.lib}/lib"
    find $out/lib -name '*.so*' ! -type l | while read -r f; do
      patchelf --set-rpath "$rpath" "$f"
    done

    install -Dm644 \
      ${finalAttrs.src}/include/parakeet_capi.h \
      $out/include/parakeet_capi.h
    runHook postInstall
  '';

  meta = {
    description = "C++/ggml inference engine for NVIDIA Parakeet ASR";
    homepage = "https://github.com/mudler/parakeet.cpp";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
})
