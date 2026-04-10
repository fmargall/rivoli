# --- Python detection ---
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# --- nanobind ---
# Has already been added with the 'ext' folder

# --- Create Python module ---
nanobind_add_module(_binding binding.cpp)

# Link against INTERFACE rivoli library
target_link_libraries(_binding PRIVATE ${PROJECT_NAME})

# Make sure public headers are seen
target_include_directories(_binding PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# OpenMP is required in parallel mode when lots
# of interpolations are requested by the user.
find_package(OpenMP REQUIRED)
target_link_libraries(_binding PRIVATE OpenMP::OpenMP_CXX)

# Copy pure Python files next to compiled module
file(GLOB PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/rivoli/*.py")

# Also add to the compiled module the bindings
set_target_properties(_binding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/rivoli
)

file(COPY ${PYTHON_FILES}
     DESTINATION ${CMAKE_BINARY_DIR}/python/rivoli)
