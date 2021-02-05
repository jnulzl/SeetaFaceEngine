cmake_minimum_required(VERSION 3.1.0)

#project(seeta_facedet_lib)

# Build options
option(BUILD_EXAMPLES   "Set to ON to build examples"    ON)
option(USE_OPENMP       "Set to ON to build use openmp"  ON)
option(USE_SSE          "Set to ON to build use SSE"     ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# Use SSE
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

# Use OpenMP
if (USE_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(include)

set(src_files 
    src/util/nms.cpp
    src/util/image_pyramid.cpp
    src/io/lab_boost_model_reader.cpp
    src/io/surf_mlp_model_reader.cpp
    src/feat/lab_feature_map.cpp
    src/feat/surf_feature_map.cpp
    src/classifier/lab_boosted_classifier.cpp
    src/classifier/mlp.cpp
    src/classifier/surf_mlp.cpp
    src/face_detection.cpp
    src/fust.cpp
    )

# Build shared library
add_library(seeta_facedet_lib SHARED ${src_files})
set(facedet_required_libs seeta_facedet_lib)

# Build examples
if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS})
        list(APPEND facedet_required_libs ${OpenCV_LIBS})
        if(OpenCV_VERSION_MAJOR VERSION_LESS 4)
            add_executable(facedet_test src/test/facedetection_test.cpp)
            target_link_libraries(facedet_test ${facedet_required_libs})
        endif()
    endif()
endif()

if(BUILD_PYTHON_API)
    set(PY_LIB_NAME seeta_facedet)
#    set(PYTHON_EXECUTABLE YOUR_PYTHON_EXECUTABLE_PATH)
    set(pybind11_DIR ${PROJECT_SOURCE_DIR}/third_party/pybind11/share/cmake/pybind11)
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