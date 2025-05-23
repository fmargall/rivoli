add_subdirectory(app)

# Add an executable for testing the interpolator
add_executable(${PROJECT_NAME}.interp headerOnly.cpp
               logger.cpp)

# Add include directories for the interpolator executable
target_include_directories(${PROJECT_NAME}.interp PRIVATE 
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/ext/glm)