cmake_minimum_required (VERSION  2.8.4)

project (SeetaFaceEngine)

# Build options
option(BUILD_PYTHON_API "Set to ON to build python api"  ON)
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)

set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin/)
file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})

add_subdirectory(FaceDetection)
add_subdirectory(FaceAlignment)
add_subdirectory(FaceIdentification)