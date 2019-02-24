#include "tensor.h"
#include "core/session/onnxruntime_c_api.h"
#include "utils.h"

Tensor Tensor::From(Napi::Value val, ONNXTensorElementDataType dataType, const char * name) {
  Napi::Env env(val.Env());
  Napi::HandleScope scope(env);

  if (!val.IsObject()) {
    Napi::Error::New(env, "tensor must be an object").ThrowAsJavaScriptException();
  }

  auto tensorObject = val.As<Napi::Object>();
  auto dimsValue = tensorObject.Get("dims");
  if (!dimsValue.IsArray()) {
    Napi::Error::New(env, "tensor.dims must be an array").ThrowAsJavaScriptException();
  }

  auto dimsArray = dimsValue.As<Napi::Array>();
  auto len = dimsArray.Length();
  std::vector<size_t> dims;
  dims.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Value dimValue = dimsArray[i];
    if (!dimValue.IsNumber()) {
      Napi::Error::New(env, "tensor.dims must be an array of numbers").ThrowAsJavaScriptException();
    }
    auto dimNumber = dimValue.As<Napi::Number>();
    double dimDouble = dimNumber.DoubleValue();
    if (floor(dimDouble) != dimDouble || dimDouble < 0 || dimDouble > 4294967295) {
      Napi::Error::New(env, "invalid dimension").ThrowAsJavaScriptException();
    }
    size_t dim = static_cast<size_t>(dimDouble);
    dims.push_back(dim);
  }

  auto dataValue = tensorObject.Get("data");
  if (!dataValue.IsTypedArray()) {
    Napi::Error::New(env, "tensor.data must be an typed array").ThrowAsJavaScriptException();
  }
  auto dataTypedArray = dataValue.As<Napi::TypedArray>();
  if (napi_float32_array != dataTypedArray.TypedArrayType() || dataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Napi::TypeError::New(env, "currently only support FLOAT32 typed array").ThrowAsJavaScriptException();
  }
  void * buffer = dataTypedArray.ArrayBuffer().Data();
  size_t byteOffset = dataTypedArray.ByteOffset();
  size_t byteLength = dataTypedArray.ByteLength();
  void * data = static_cast<char *>(buffer) + byteOffset;
  
  return {data, byteLength, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims, name};
}

Napi::Value Tensor::ToNapiValue(napi_env env) {
  Napi::EscapableHandleScope scope(env);

  auto shapeArray = CreateNapiArrayFrom(env, this->shape);

  auto buffer = Napi::ArrayBuffer::New(env, this->dataLength);
  memcpy(buffer.Data(), this->data, this->dataLength);

  if (this->type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Napi::TypeError::New(env, "currently only support FLOAT32 typed array").ThrowAsJavaScriptException();
  }
  auto data = Napi::Float32Array::New(env, dataLength / 4, buffer, 0);

  auto tensor = Napi::Object::New(env);
  tensor.Set("dims", shapeArray);
  tensor.Set("data", data);
  tensor.Set("type", Napi::Number::New(env, static_cast<double>(this->type)));

  return scope.Escape(tensor);
}
