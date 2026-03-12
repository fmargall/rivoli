include(FetchContent)

# nanobind Python binding library
add_subdirectory(nanobind)

# vectra SIMD library
FetchContent_Declare(
    vectra
    GIT_REPOSITORY https://github.com/fmargall/vectra.git
    GIT_TAG main
)

FetchContent_MakeAvailable(vectra)

# Eigen library
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)