cmake_minimum_required(VERSION 3.10)

project(ANN-c C)

include_directories(include)
find_package(OpenMP REQUIRED)

option(BUILD_TRAIN "Build doodle_train main" ON)
option(BUILD_TRAIN_OMP "Build doodle_train_omp main" ON)
option(BUILD_XOR "Build xor_sigmoid main" OFF)

if(BUILD_TRAIN)
    add_executable(doodle_train 
        src/doodle_train.c
    )

    target_link_libraries(doodle_train m OpenMP::OpenMP_C)

    add_custom_command(TARGET doodle_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()

if(BUILD_XOR)
    add_executable(xor_main 
        src/xor_sigmoid.c
    )

    target_link_libraries(xor_main m OpenMP::OpenMP_C)

    add_custom_command(TARGET xor_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()

if(BUILD_TRAIN_OMP)
    add_executable(doodle_train_omp 
        src/doodle_train_omp.c
    )

    target_link_libraries(doodle_train_omp m OpenMP::OpenMP_C)

    add_custom_command(TARGET doodle_train_omp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()
