enable_testing()

include(FetchContent)

# Download GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
)

# Disable unnecessary parts
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(RIVOLI.tests TestMain.cpp TestArrays.cpp TestCoordinateConversion.cpp
               ${CMAKE_SOURCE_DIR}/src/logger.cpp)

# Include directories
target_include_directories(RIVOLI.tests PRIVATE
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/ext/glm
                           ${CMAKE_SOURCE_DIR}/ext/eigen)
# Link libraries
target_link_libraries(RIVOLI.tests
  PRIVATE
    gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(RIVOLI.tests)
