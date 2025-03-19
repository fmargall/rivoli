cmake_minimum_required(VERSION 3.10)
project(RIVOLI)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenMP required for parallel computing
find_package(OpenMP REQUIRED)

# Add the main project for the executable
add_executable(RIVOLI.App src/main.cpp 
               src/arrays.hpp
               src/configurator.hpp src/configurator.cpp
               src/coordinate.hpp src/coordinateConversion.hpp
               src/interpolator.hpp src/interpolator.cpp
               src/logger.hpp src/logger.cpp
               src/topology.hpp
               src/export.hpp)

target_link_libraries(RIVOLI.App PUBLIC OpenMP::OpenMP_CXX)

# Set RIVOLI.App as start-up project in Visual Studio 
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT RIVOLI.App)

# Exterior submodules
target_include_directories(RIVOLI.App PRIVATE ext/glm ext/eigen)

# Set the command arguments for the Visual Studio debugger
set_target_properties(RIVOLI.App PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "../resources/config.ini")

# Remove the ZERO_CHECK project
set(CMAKE_SUPPRESS_REGENERATION true)