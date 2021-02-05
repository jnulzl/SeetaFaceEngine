cmake_minimum_required(VERSION 2.8.4)

project(seeta_fa_lib)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)

# Use C++11
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "C++11 support has been enabled by default.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

include_directories(include)

set(src_files 
    src/cfan.cpp
    src/face_alignment.cpp
    src/sift.cpp
    )

add_library(seeta_fa_lib SHARED ${src_files})
set(fa_required_libs seeta_fa_lib)

if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS} build)
	    link_directories(build)
        list(APPEND fa_required_libs ${OpenCV_LIBS} seeta_facedet_lib)
        if(OpenCV_VERSION_MAJOR VERSION_LESS 4)
            add_executable(fa_test src/test/face_alignment_test.cpp)
            target_link_libraries(fa_test ${fa_required_libs})
        endif()
    endif()
endif()

if(BUILD_PYTHON_API)
    set(PY_LIB_NAME seeta_fa)
    #    set(PYTHON_EXECUTABLE YOUR_PYTHON_EXECUTABLE_PATH)
    set(pybind11_DIR ${PROJECT_SOURCE_DIR}/../third_party/pybind11/share/cmake/pybind11)
    find_package(pybind11 REQUIRED)

    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)
    set(src_for_python ${src_for_python} ${_tmp})
    pybind11_add_module(${PY_LIB_NAME}_python ${src_for_python})
    target_link_libraries(${PY_LIB_NAME}_python PRIVATE
            ${PY_LIB_NAME}_lib)
    target_include_directories(${PY_LIB_NAME}_python PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            )
    target_compile_options(${PY_LIB_NAME}_python PRIVATE -DMODEL_NAME=${PY_LIB_NAME})
    set_target_properties(${PY_LIB_NAME}_python PROPERTIES OUTPUT_NAME ${PY_LIB_NAME})
endif()