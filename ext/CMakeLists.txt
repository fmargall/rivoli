include(FetchContent)

# vectra SIMD library
FetchContent_Declare(
    vectra
    GIT_REPOSITORY https://github.com/fmargall/vectra.git
    GIT_TAG main
)

FetchContent_MakeAvailable(vectra)