# Adding the library
add_library(${PROJECT_NAME}.lib STATIC)

# Adding the source files
target_sources(${PROJECT_NAME}.lib PRIVATE Lib.cpp)