cmake_minimum_required(VERSION 3.21)

# Get automatic project name, version and description
# directly from the Git repository using git describe
get_filename_component(PROJECT_NAME_FROM_DIR 
    ${CMAKE_CURRENT_SOURCE_DIR} NAME
)

# The following part allows us to get the version
# from git tags, and if none, fallbacks to 0.0.0.
execute_process(
    COMMAND git describe --tags --dirty --always
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Fallback if version fails
set(PROJECT_VERSION      "0.0.0")
set(PROJECT_VERSION_FULL "0.0.0")

if(GIT_DESCRIBE AND NOT GIT_DESCRIBE STREQUAL "")
    set(PROJECT_VERSION_FULL ${GIT_DESCRIBE})

    # Extract strict semver part only
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION_MATCH ${GIT_DESCRIBE})
    if(VERSION_MATCH)
        set(PROJECT_VERSION ${VERSION_MATCH})
    endif()
endif()

project(${PROJECT_NAME_FROM_DIR}
    VERSION ${PROJECT_VERSION}
    LANGUAGES CXX
)

# Let's add the main library target
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(${PROJECT_NAME}
    INTERFACE
        vectra
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        eigen
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        tinylogger
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.hpp
    @ONLY
)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Build tests if requested
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Link each dependency
add_subdirectory(ext)

# Build benchmarks if requested
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# Build playgrounds if requested
option(BUILD_PLAYGROUND "Build playground" OFF)
if(BUILD_PLAYGROUND)
    add_subdirectory(playground)
endif()

# Build Python binding if requested
option(BUILD_BINDING "Build binding" ON)
if(BUILD_BINDING)
    add_subdirectory(python)
endif()
