if(BUILD_EXAMPLES)
  #aux_source_directory (. SRC_LIST)
  set(SRC_LIST
          test_face_recognizer.cpp
          test_face_verification.cpp
          )

  message(${SRC_LIST})

  enable_testing ()
  # add external libraries
  find_package(OpenCV REQUIRED)
  message(STATUS "OpenCV Version: " ${OpenCV_VERSION})
  if(OpenCV_VERSION_MAJOR VERSION_LESS 4)
    include_directories(${PROJECT_SOURCE_DIR}/FaceDetection/include)
    include_directories(${PROJECT_SOURCE_DIR}/FaceAlignment/include)
    include_directories(${PROJECT_SOURCE_DIR}/FaceIdentification/include)
    foreach (f ${SRC_LIST})
      string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
      add_executable(${BIN} ${f})
      target_link_libraries(${BIN} viplnet ${OpenCV_LIBS})
    endforeach ()
  endif()
endif()