# Adding the executable
add_executable(${PROJECT_NAME}.app Main.cpp
               ${CMAKE_SOURCE_DIR}/src/logger.cpp)

# Add RIVOLI icon for the Windows executable
if(WIN32)
    target_sources(${PROJECT_NAME}.app PRIVATE ${CMAKE_SOURCE_DIR}/src/app/resources.rc)

    # Cpy icon into build folder in order for rc.exe to find it
    add_custom_command(TARGET RIVOLI.app PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/resources/RIVOLI.ico
        ${CMAKE_CURRENT_BINARY_DIR}/resources/RIVOLI.ico
    )
endif()

target_sources(${PROJECT_NAME}.app PRIVATE
    BrdfSamplesCoupole.cpp
)

target_include_directories(${PROJECT_NAME}.app PRIVATE 
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/ext/glm
                           ${CMAKE_SOURCE_DIR}/ext/eigen)