set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)

add_executable(glsl-decompiler main.cpp)

create_target_directory_groups(glsl-decompiler)

target_link_libraries(glsl-decompiler PRIVATE video_core)

if(UNIX AND NOT APPLE)
    install(TARGETS glsl-decompiler RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()
