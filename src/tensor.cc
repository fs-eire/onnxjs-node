#include <cmath>
#include <memory>
#include <sstream>

#include "onnxruntime_c_api.h"
#include "tensor.h"
#include "utils.h"

// make sure consistent with origin definition
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == 0, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == 1, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 == 2, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 == 3, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 == 4, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 == 5, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 == 6, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 == 7, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING == 8, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL == 9, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 == 10, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE == 11, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 == 12, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 == 13, "definition not consistent with OnnxRuntime");
const size_t ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT = 14;

// size of element in bytes for each data type. 0 indicates not supported.
const size_t DATA_TYPE_ELEMENT_SIZE_MAP[] = {
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not supported
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    2, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    2, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64         INT64 not working in
       // Javascript
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        not supported
    1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    2, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16       FLOAT16 not working in
       // Javascript
    8, // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    4, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64        UINT64 not working in
       // Javascript
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not supported
    0, // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not supported
    0  // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not supported
};

const napi_typedarray_type DATA_TYPE_TYPEDARRAY_MAP[] = {
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not
                        // supported
    napi_float32_array, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    napi_int8_array,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    napi_uint16_array,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    napi_int16_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    napi_int32_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64         INT64 not
                        // working i
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        not
                        // supported
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    napi_int8_array,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16       FLOAT16 not
                        // working
    napi_float64_array, // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    napi_uint32_array,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64        UINT64 not
                        // working
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not
                        // supported
    napi_uint8_array,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not
                        // supported
    napi_uint8_array    // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not
                        // supported
};

Tensor Tensor::From(Napi::Value val, const char *name) {
  Napi::Env env(val.Env());
  Napi::HandleScope scope(env);

  if (!val.IsObject()) {
    throw Napi::TypeError::New(env, "tensor must be an object");
  }

  auto tensorObject = val.As<Napi::Object>();
  auto dimsValue = tensorObject.Get("dims");
  if (!dimsValue.IsArray()) {
    throw Napi::TypeError::New(env, "tensor.dims must be an array");
  }

  auto dimsArray = dimsValue.As<Napi::Array>();
  auto len = dimsArray.Length();
  std::vector<size_t> dims;
  dims.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Value dimValue = dimsArray[i];
    if (!dimValue.IsNumber()) {
      throw Napi::TypeError::New(env, "tensor.dims must be an array of numbers");
    }
    auto dimNumber = dimValue.As<Napi::Number>();
    double dimDouble = dimNumber.DoubleValue();
    if (std::floor(dimDouble) != dimDouble || dimDouble < 0 || dimDouble > 4294967295) {
      throw Napi::TypeError::New(env, "invalid dimension");
    }
    size_t dim = static_cast<size_t>(dimDouble);
    dims.push_back(dim);
  }

  auto dataValue = tensorObject.Get("data");
  if (!dataValue.IsTypedArray()) {
    throw Napi::TypeError::New(env, "tensor.data must be an typed array");
  }
  auto dataTypedArray = dataValue.As<Napi::TypedArray>();

  auto dataTypeValue = tensorObject.Get("type");
  if (!dataTypeValue.IsNumber()) {
    throw Napi::TypeError::New(env, "tensor.type must be a number");
  }
  auto dataTypeNumber = dataTypeValue.As<Napi::Number>();
  auto dataTypeDouble = dataTypeNumber.DoubleValue();
  if (std::floor(dataTypeDouble) != dataTypeDouble || dataTypeDouble < 0 ||
      dataTypeDouble > ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT) {
    throw Napi::TypeError::New(env, "tensor.type must be a valid integer");
  }
  auto dataType = static_cast<ONNXTensorElementDataType>(static_cast<size_t>(dataTypeDouble));

  if (DATA_TYPE_TYPEDARRAY_MAP[dataType] != dataTypedArray.TypedArrayType()) {
    throw Napi::TypeError::New(env, "tensor.type does not match the type of tensor.data");
  }

  void *buffer = dataTypedArray.ArrayBuffer().Data();
  size_t byteOffset = dataTypedArray.ByteOffset();
  size_t byteLength = dataTypedArray.ByteLength();
  void *data = static_cast<char *>(buffer) + byteOffset;

  Tensor t;
  t.data = data;
  t.dataLength = byteLength;
  t.type = dataType;
  t.shape = dims;
  t.name = name;
  return t;
}

Tensor Tensor::From(OrtValue *value, const char *name) {
  if (!OrtIsTensor(value)) {
    throw std::runtime_error("Unsupported value type. Only Tensor value is supported.");
  }

  void *data;
  auto status = OrtGetTensorMutableData(value, &data);
  if (status) {
    std::ostringstream what;
    what << "Failed to get data from output tensors: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }

  OrtTensorTypeAndShapeInfo *tensorInfo;
  status = OrtGetTensorShapeAndType(value, &tensorInfo);
  if (status) {
    std::ostringstream what;
    what << "Failed to get tensor info from output tensors: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }
  std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(&OrtReleaseTensorTypeAndShapeInfo)> tensorInfoScopeGuard(
      tensorInfo, OrtReleaseTensorTypeAndShapeInfo);

  auto dataType = OrtGetTensorElementType(tensorInfo);
  auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
  std::vector<int64_t> dims(dimsCount);
  if (dimsCount > 0) {
    OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
  }
  auto elementCount = OrtGetTensorShapeElementCount(tensorInfo);
  if (elementCount <= 0) {
    std::ostringstream what;
    what << "Invalid element count (" << elementCount << ")";
    throw std::runtime_error(what.str());
  }

  auto byteLength = static_cast<size_t>(elementCount) * DATA_TYPE_ELEMENT_SIZE_MAP[dataType];
  if (byteLength == 0) {
    std::ostringstream what;
    what << "unsupported data type (" << dataType << ")";
    throw std::runtime_error(what.str());
  }

  std::vector<size_t> dimsUnsigned(dimsCount);
  for (size_t i = 0; i < dimsCount; i++) {
    dimsUnsigned[i] = static_cast<size_t>(dims[i]);
  }

  Tensor t;
  t.data = data;
  t.dataLength = byteLength;
  t.type = dataType;
  t.shape = dimsUnsigned;
  t.value = value;
  return t;
}

Napi::Value Tensor::ToNapiValue(napi_env env) {
  Napi::EscapableHandleScope scope(env);

  auto shapeArray = CreateNapiArrayFrom(env, this->shape);

  auto buffer = Napi::ArrayBuffer::New(env, this->dataLength);
  memcpy(buffer.Data(), this->data, this->dataLength);

  if (this->type < 0 || this->type > ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT) {
    throw Napi::TypeError::New(env, "invalid data type detected");
  }

  size_t elementSize = DATA_TYPE_ELEMENT_SIZE_MAP[this->type];
  Napi::TypedArray data;
  switch (this->type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    data = Napi::Float32Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    data = Napi::Uint8Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    data = Napi::Int8Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    data = Napi::Uint16Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    data = Napi::Int16Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    data = Napi::Int32Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    data = Napi::Uint8Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    data = Napi::Float64Array::New(env, dataLength / elementSize, buffer, 0);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    data = Napi::Uint32Array::New(env, dataLength / elementSize, buffer, 0);
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default:
    throw Napi::TypeError::New(env, "unsupported data type detected");
    break;
  }
  if (DATA_TYPE_TYPEDARRAY_MAP[this->type] != data.TypedArrayType()) {
    throw Napi::TypeError::New(env, "mismatched typed array type");
  }

  auto tensor = Napi::Object::New(env);
  tensor.Set("dims", shapeArray);
  tensor.Set("data", data);
  tensor.Set("type", Napi::Number::New(env, static_cast<double>(this->type)));

  return scope.Escape(tensor);
}
