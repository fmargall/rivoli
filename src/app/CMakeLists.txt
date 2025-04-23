# Adding the executable
add_executable(${PROJECT_NAME}.app Main.cpp
               ${CMAKE_SOURCE_DIR}/src/logger.cpp)

target_include_directories(${PROJECT_NAME}.app PRIVATE 
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/ext/glm
                           ${CMAKE_SOURCE_DIR}/ext/eigen)