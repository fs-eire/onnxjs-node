#ifndef ONNXRUNTIME_NODE_TENSOR_H
#define ONNXRUNTIME_NODE_TENSOR_H

#pragma once

#include <vector>
#include <napi.h>

#include "core/session/onnxruntime_c_api.h"

// a simple structure that represents a tensor.
//
// NOTES: this type is not responsible for memory management of all its fields.
//        it always assume the memory is available for field 'data' and 'name'
struct Tensor {
    // pointer to the raw data
    void * data;

    // data length in bytes
    size_t dataLength;

    // specify the data type
    ONNXTensorElementDataType type;

    // specify the tensor's shape 
    std::vector<size_t> shape;

    // specify the tensor's name. This is optional.
    const char * name;

    // the temporary underlying OrtValue pointer.
    OrtValue * value;

    static Tensor From(Napi::Value val, const char * name = nullptr);
    static Tensor From(OrtValue *value, const char * name = nullptr);
    Napi::Value ToNapiValue(napi_env env);
};

#endif
