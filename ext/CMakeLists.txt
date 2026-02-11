include(FetchContent)

# vectra SIMD library
FetchContent_Declare(
    vectra
    GIT_REPOSITORY https://github.com/fmargall/vectra.git
)

FetchContent_MakeAvailable(vectra)