add_subdirectory(app)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.hpp
)

# Add an executable for testing the interpolator
add_executable(${PROJECT_NAME}.interp headerOnly.cpp
               logger.cpp)

# Add include directories for the interpolator executable
target_include_directories(${PROJECT_NAME}.interp PRIVATE 
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/ext/glm)