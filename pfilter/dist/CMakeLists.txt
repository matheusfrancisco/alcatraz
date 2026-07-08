# Builds a single self-contained shared libpf (libpf.dylib / libpf.so) from a
# privacy-filter.cpp checkout, for the purego binding in this module.
#
# Upstream builds pf as a STATIC library (and ggml as shared libs by
# default). A dlopen-based binding wants the opposite: one relocatable shared
# object that exports the pf_* symbols and carries ggml inside it, so a
# single file is all a user ever has to install. This wrapper compiles
# everything PIC, links ggml statically, and force-loads the pf archive into
# a SHARED target (nothing references its symbols at link time, so a plain
# link would drop them all).
#
#   git clone --recursive https://github.com/localai-org/privacy-filter.cpp
#   cmake -S pfilter/dist -B build \
#         -DPF_SOURCE_DIR=/path/to/privacy-filter.cpp \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   # -> build/libpf.dylib (macOS) or build/libpf.so (Linux)
#
# Flags worth knowing (all passed straight through to ggml):
#   -DGGML_NATIVE=OFF   portable build (no -march=native); what CI ships.
#                       Add -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON on
#                       x86-64 for a sane portable baseline.
#   -DPF_CUDA=ON / -DPF_VULKAN=ON   GPU backends (build on a machine with
#                       the toolkit; the distributed artifacts are CPU-only).
cmake_minimum_required(VERSION 3.21)
project(libpf-dist C CXX)

set(PF_SOURCE_DIR "" CACHE PATH "privacy-filter.cpp checkout (ggml submodule initialized)")
if(NOT PF_SOURCE_DIR)
    message(FATAL_ERROR "Set -DPF_SOURCE_DIR=/path/to/privacy-filter.cpp")
endif()

# One self-contained artifact: ggml enters libpf statically, and every object
# is PIC so the static archives can be folded into a shared library.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PF_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(PF_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(${PF_SOURCE_DIR} pf-build)

add_library(pf_shared SHARED shim.cpp)
set_target_properties(pf_shared PROPERTIES OUTPUT_NAME pf)

# The pf_* symbols live in the static pf archive and nothing in shim.cpp
# references them — force the whole archive into the shared object.
if(APPLE)
    target_link_options(pf_shared PRIVATE "-Wl,-force_load,$<TARGET_FILE:pf>")
else()
    target_link_options(pf_shared PRIVATE
        "-Wl,--whole-archive,$<TARGET_FILE:pf>,--no-whole-archive")
endif()
# Regular link too: ordering dependency + transitive static ggml libs.
target_link_libraries(pf_shared PRIVATE pf)
