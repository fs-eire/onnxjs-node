cmake_minimum_required(VERSION 3.11)

project (onnxjs-node)

set(CMAKE_CXX_STANDARD 14)

message(${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_JS_INC})
include_directories(${CMAKE_SOURCE_DIR}/onnxruntime/inc)
include_directories(${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

# CPU
add_library(onnxruntime SHARED ${SOURCE_FILES})
set_target_properties(onnxruntime PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(onnxruntime ${CMAKE_JS_LIB})
if (WIN32)
  target_link_libraries(onnxruntime ${CMAKE_SOURCE_DIR}/onnxruntime/bin/win-x64/native/onnxruntime.lib)
endif()
