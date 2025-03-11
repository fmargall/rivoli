cmake_minimum_required(VERSION 3.10)
project(RIVOLI)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the main project for the executable
add_executable(RIVOLI src/main.cpp)
# Exterior submodules
target_include_directories(RIVOLI PRIVATE ext/glm ext/eigen)

# Set the command arguments for the Visual Studio debugger
set_target_properties(RIVOLI PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "resources/config.ini")

# Remove the ZERO_CHECK project
set(CMAKE_SUPPRESS_REGENERATION true)