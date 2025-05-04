cmake_minimum_required(VERSION 3.10)

project(ANN-c C)

include_directories(include)

cmake_minimum_required(VERSION 3.10)

project(ANN-c C)

include_directories(include)

option(BUILD_MAIN "Build doodle_train main" OFF)
option(BUILD_XOR "Build xor_sigmoid main" OFF)
option(BUILD_DIGIT_CLASSIFIER "Build digit_classifier main" ON)

if(BUILD_MAIN)
    add_executable(doodle_train 
        src/doodle_train.c
    )

    target_link_libraries(doodle_train m)

    add_custom_command(TARGET doodle_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()

if(BUILD_XOR)
    add_executable(xor_main 
        src/xor_sigmoid.c
    )

    target_link_libraries(xor_main m)

    add_custom_command(TARGET xor_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()


if(BUILD_DIGIT_CLASSIFIER)
    add_executable(digit_classifier 
        src/digit_classifier.c
    )

    target_link_libraries(digit_classifier m)

    add_custom_command(TARGET digit_classifier POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/datasets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/datasets ${CMAKE_BINARY_DIR}/datasets
    )
endif()