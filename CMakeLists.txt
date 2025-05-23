cmake_minimum_required(VERSION 3.21)

# Set the project name and version. While the project name should not be modified,
# the version should be updated with each release, later automatically using CI/CD
project(RIVOLI VERSION 0.6.1 LANGUAGES CXX)

# Used for parallel computing
find_package(OpenMP REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Will generate debug symbols (even in Release mode) for profiling
if(MSVC)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif()

# Use Address Sanitizer for debugging. Should be removed for final Release
#add_compile_options(/fsanitize=address)
#add_link_options(/INFERASANLIBS)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
