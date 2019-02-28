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
set_target_properties(onnxruntime PROPERTIES
  PREFIX "" SUFFIX ".node"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH FALSE
)
target_link_libraries(onnxruntime PRIVATE ${CMAKE_JS_LIB})
if (WIN32)
  target_link_libraries(onnxruntime PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime/bin/win-x64/onnxruntime.lib)
elseif (APPLE)
  target_link_libraries(onnxruntime PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime/bin/darwin-x64/libonnxruntime.0.2.1.dylib)
  set_target_properties(onnxruntime PROPERTIES INSTALL_RPATH "@loader_path")
else()
  target_link_libraries(onnxruntime PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime/bin/linux-x64/libonnxruntime.so.0.2.1)
  set_target_properties(onnxruntime PROPERTIES INSTALL_RPATH "$ORIGIN/")
endif()

# GPU
if (NOT APPLE)
  add_library(onnxruntime_gpu SHARED ${SOURCE_FILES})
  set_target_properties(onnxruntime_gpu PROPERTIES
    PREFIX "" SUFFIX ".node"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
  target_link_libraries(onnxruntime_gpu PRIVATE ${CMAKE_JS_LIB})
  if (WIN32)
    target_link_libraries(onnxruntime_gpu PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime/bin/win_gpu-x64/onnxruntime.lib)
  else()
    target_link_libraries(onnxruntime_gpu PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime/bin/linux_gpu-x64/libonnxruntime.so.0.2.1)
    set_target_properties(onnxruntime_gpu PROPERTIES INSTALL_RPATH "$ORIGIN/")
  endif()
endif()
