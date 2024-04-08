cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(MAIN_TARGET         rf_anomaly)
project(${MAIN_TARGET})

###################################################################
set(PLATFORM_DIR        ${CMAKE_CURRENT_LIST_DIR})
set(VENDOR_DIR          ${PLATFORM_DIR}/vendor)

SET(CFG_TENSORFLOW_VER      "v2.16.1" CACHE STRING "Tensorflow Git Tag Version")
SET(CFG_TENSORFLOW_REPO     "git@github.com:tensorflow/tensorflow.git")
set(TENSORFLOW_SOURCE_DIR   ${VENDOR_DIR}/tensorflow_src)
###################################################################
message(STATUS "Fething code dependencies ...")
FetchContent_Declare(tensorflow_repo
    SOURCE_DIR              ${VENDOR_DIR}/tensorflow_src
    GIT_REPOSITORY          ${CFG_TENSORFLOW_REPO}
    GIT_TAG                 ${CFG_TENSORFLOW_VER}
    UPDATE_DISCONNECTED     ON
)
FetchContent_MakeAvailable(tensorflow_repo)

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

###################################################################
message(STATUS "Building model deployment code ...")
set(SOURCES
    rf_anomaly.cc
)

set(INCLUDE_DIRS 
    include
)

# executable target
add_executable(${MAIN_TARGET} ${SOURCES})

# Include directories
target_include_directories(${MAIN_TARGET} PUBLIC ${INCLUDE_DIRS})

###################################################################
message(STATUS "Linking Libraries ... ")

# Link TensorFlow Lite to the executable target
target_link_libraries(${MAIN_TARGET} tensorflow-lite)
