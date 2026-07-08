// Intentionally empty: gives the pf_shared target a source file (CMake
// requires one for SHARED libraries) and makes it link as C++, pulling in
// the C++ runtime that the static pf/ggml objects need. The pf_* symbols
// themselves are force-loaded from the static archive — see CMakeLists.txt.
