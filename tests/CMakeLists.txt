enable_testing()

# Adding Google Test subdirectory
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/googletest ${CMAKE_BINARY_DIR}/ext/googletest)

# Add executable for testing
add_executable(${PROJECT_NAME}.tests TestMain.cpp)

# Link Google Test library
target_link_libraries(${PROJECT_NAME}.tests gtest gtest_main ${PROJECT_NAME}.lib)

# Adding tests
add_test(NAME TestMain COMMAND ${PROJECT_NAME}.tests)