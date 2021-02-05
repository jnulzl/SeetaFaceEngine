cmake_minimum_required (VERSION  2.8.4)

#project (viplnet)


set (VIPLNET_VERSION_MAJOR 4)
set (VIPLNET_VERSION_MINOR 5)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGTEST_USE_OWN_TR1_TUPLE=1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGTEST_USE_OWN_TR1_TUPLE=1")
endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

set(VIPLNET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VIPLNET_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# set __VIOL_LOG__ macro
# add_definitions(-D__VIPL_LOG__)

include_directories(${VIPLNET_INCLUDE_DIR})
include_directories(${VIPLNET_SRC_DIR})

add_subdirectory(src)

aux_source_directory(./src SRC_LIST)
aux_source_directory(./tools TOOLS_LIST)
add_library(viplnet SHARED ${SRC_LIST} ${TOOLS_LIST})
set_target_properties(viplnet PROPERTIES 
  VERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR}
  SOVERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR})

if(BUILD_PYTHON_API)
    set(PY_LIB_NAME viplnet)
    #    set(PYTHON_EXECUTABLE YOUR_PYTHON_EXECUTABLE_PATH)
    set(pybind11_DIR ${PROJECT_SOURCE_DIR}/third_party/pybind11/share/cmake/pybind11)
    find_package(pybind11 REQUIRED)

    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)
    set(src_for_python ${src_for_python} ${_tmp})
    pybind11_add_module(${PY_LIB_NAME}_python ${src_for_python})
    target_link_libraries(${PY_LIB_NAME}_python PRIVATE
            ${PY_LIB_NAME})
    target_include_directories(${PY_LIB_NAME}_python PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            )
    target_compile_options(${PY_LIB_NAME}_python PRIVATE -DMODEL_NAME=${PY_LIB_NAME})
    set_target_properties(${PY_LIB_NAME}_python PROPERTIES OUTPUT_NAME ${PY_LIB_NAME})
endif()