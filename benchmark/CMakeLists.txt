include(FetchContent)

# Download Google Benchmark
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)

# Disable benchmark tests
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# List all benchmark sources
file(GLOB BENCH_SOURCES
     CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${PROJECT_NAME}_benchmark ${BENCH_SOURCES})

target_link_libraries(${PROJECT_NAME}_benchmark
    PRIVATE
        ${PROJECT_NAME}
        benchmark::benchmark
)