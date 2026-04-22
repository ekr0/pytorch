// CPU AOTI Triton runtime helpers.
//
// Mirrors `sycl_runtime_wrappers.h` (XPU) and `loadKernel` in
// `static_launcher/cuda.cpp`. The generated CPU AOTI wrapper calls the helpers
// here to `dlopen` the per-kernel `.so` and the per-kernel launcher `.so`,
// and resolve the kernel symbol + the launcher entry point.
//
// The `soDir` override mirrors GPU's `binDir` / `cubin_dir_` -- when an AOTI
// package is extracted to a directory other than where it was built, the
// runtime can pass an override and we resolve `<soDir>/<basename>` in place
// of the absolute path baked into the wrapper at compile time.
//
// Launcher ABI: the helper expects the launcher `.so` to export a C symbol
// named `run_from_nativert` with the signature below. This symbol is produced
// by Triton CPU backends configured to emit a NativeRT-compatible launcher
// (e.g. via the `make_cpp_launcher` helper). If your Triton CPU build does
// not export this symbol, `loadCpuTritonLauncher` aborts with a clear error
// at AOTI model load time -- this codepath is unsupported in that case.

#pragma once

#ifdef _WIN32
#error "CPU AOTI Triton runtime helpers are not supported on Windows"
#endif

#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <filesystem>
#include <optional>
#include <string>

namespace torch::aot_inductor {

// Signature of the launcher `run_from_nativert` symbol expected in every
// Triton CPU launcher `.so` we load. (gridX, gridY, gridZ, num_cpu_threads,
// void** args, kernel_fn_ptr).
using CpuTritonLauncherFn =
    void (*)(uint32_t, uint32_t, uint32_t, int, void**, void*);

// Resolve `filePath` against `soDir` if provided -- replaces the directory
// part with `soDir` while keeping the basename. Mirrors the pattern in
// `sycl_runtime_wrappers.h::loadKernel` and
// `static_launcher/cuda.cpp::loadKernel`.
//
// If `soDir` is empty AND the absolute `filePath` doesn't exist anymore (e.g.
// the AOTI package was extracted into a tempdir without anyone passing
// `cubin_dir_`), fall back to looking up the basename next to the AOTI .so
// itself via `dladdr`. This lets the wrapper find its kernel/launcher .so
// files when the runtime forgets to pass an explicit override.
[[maybe_unused]] static std::string _resolve_cpu_triton_so_path(
    std::string filePath,
    const std::optional<std::string>& soDir) {
  if (soDir) {
    std::filesystem::path p1{*soDir};
    std::filesystem::path p2{filePath};
    return (p1 / p2.filename()).string();
  }
  std::error_code ec;
  if (std::filesystem::exists(filePath, ec)) {
    return filePath;
  }
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&_resolve_cpu_triton_so_path), &info) &&
      info.dli_fname != nullptr) {
    std::filesystem::path basename = std::filesystem::path(filePath).filename();
    std::filesystem::path candidate =
        std::filesystem::path(info.dli_fname).parent_path() / basename;
    if (std::filesystem::exists(candidate, ec)) {
      return candidate.string();
    }
  }
  return filePath;
}

// CPU Cpp Wrapper API: dlopen `filePath`, dlsym `funcName`, return the
// resulting function pointer. Throws via TORCH_CHECK on failure. The handle is
// intentionally leaked -- the AOTI process keeps the .so loaded for its
// lifetime, mirroring how cubin modules are leaked on the GPU side.
[[maybe_unused]] static void* loadCpuTritonKernel(
    std::string filePath,
    const std::string& funcName,
    const std::optional<std::string>& soDir = std::nullopt) {
  filePath = _resolve_cpu_triton_so_path(std::move(filePath), soDir);
  void* handle = dlopen(filePath.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "dlopen ", filePath, ": ", dlerror());
  void* fn = dlsym(handle, funcName.c_str());
  TORCH_CHECK(fn, "dlsym ", funcName, " from ", filePath, ": ", dlerror());
  return fn;
}

// CPU Cpp Wrapper API: dlopen the Triton CPU launcher `.so` and dlsym its
// `run_from_nativert` entry point. Returns the typed launcher fn pointer.
// Same handle-leak policy as `loadCpuTritonKernel` above.
//
// Aborts (TORCH_CHECK) if the symbol is not present -- this is the
// expected outcome when the Triton CPU backend in use was not built to
// emit a NativeRT-style launcher. Loading must fail loudly here so callers
// know the AOTI artifact is incompatible with the runtime Triton build.
[[maybe_unused]] static CpuTritonLauncherFn loadCpuTritonLauncher(
    std::string filePath,
    const std::optional<std::string>& soDir = std::nullopt) {
  filePath = _resolve_cpu_triton_so_path(std::move(filePath), soDir);
  void* handle = dlopen(filePath.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "dlopen ", filePath, ": ", dlerror());
  void* fn = dlsym(handle, "run_from_nativert");
  TORCH_CHECK(
      fn,
      "Triton CPU launcher .so does not export 'run_from_nativert' (",
      filePath,
      "): ",
      dlerror(),
      ". The CPU AOTI Triton path requires a Triton CPU build that emits a "
      "NativeRT-compatible launcher.");
  return reinterpret_cast<CpuTritonLauncherFn>(fn);
}

} // namespace torch::aot_inductor
