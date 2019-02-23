#include <core/providers/cpu/cpu_provider_factory.h>

#include "inference-session.h"

Napi::FunctionReference InferenceSession::constructor;

Napi::Object InferenceSession::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "InferenceSession", {
    InstanceMethod("loadModel", &InferenceSession::LoadModel),
    InstanceMethod("run", &InferenceSession::Run),
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("InferenceSession", func);
  return exports;
}

InferenceSession::InferenceSession(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferenceSession>(info)  {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  auto status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxjs", &this->env_);
  if (status) {
    Napi::Error::New(env, "Failed to create onnxruntime environment").ThrowAsJavaScriptException();
  }
  this->sessionOptions_ = OrtCreateSessionOptions();
}



Napi::Value InferenceSession::LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  size_t length = info.Length();
  if (length <= 0 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expect argument: model path").ThrowAsJavaScriptException();
  }

  Napi::String value = info[0].As<Napi::String>();
  auto status = OrtCreateSession(
    this->env_,
#ifdef _WIN32
    reinterpret_cast<const wchar_t *>(value.Utf16Value().c_str()),
#else
    value.Utf8Value().c_str(),
#endif
    this->sessionOptions_,
    &this->session_);
  if (status) {
    Napi::Error::New(env, "Failed to load model").ThrowAsJavaScriptException();
  }

  return env.Undefined();
}

OrtValue * JavascriptTensorToOrtValue(OrtAllocatorInfo *allocatorInfo, Napi::Env env, Napi::Value val) {
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
  if (napi_float32_array != dataTypedArray.TypedArrayType()) {
    Napi::Error::New(env, "currently only support FLOAT32 typed array").ThrowAsJavaScriptException();
  }
  void * buffer = dataTypedArray.ArrayBuffer().Data();
  size_t byteOffset = dataTypedArray.ByteOffset();
  size_t byteLength = dataTypedArray.ByteLength();
  void * data = static_cast<char *>(buffer) + byteOffset;
  
  OrtValue *ortTensorValue;
  OrtCreateTensorWithDataAsOrtValue(allocatorInfo, data, byteLength, &dims[0], dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortTensorValue);

  return ortTensorValue;
}

Napi::Value OrtValueToJavascriptTensor(Napi::Env env, OrtValue *value) {
  Napi::EscapableHandleScope scope(env);
  auto obj = Napi::Object::New(env);
  return scope.Escape(obj);
}


Napi::Value InferenceSession::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  // TODO: implement session.run()


  OrtAllocatorInfo *allocatorInfo;
  OrtCreateCpuAllocatorInfo(OrtDeviceAllocator, OrtMemTypeDefault, &allocatorInfo);
  // TODO: scope guard for `allocatorInfo`


  //
  // info[0]: inputs                 Tensor[]
  if (info.Length() <= 0) {
    Napi::TypeError::New(env, "Expect argument: input tensors").ThrowAsJavaScriptException();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(env, "Expect the first argument to be an array of input tensors").ThrowAsJavaScriptException();
  }
  auto passedInInputTensors = info[0].As<Napi::Array>();
  auto passedInInputsCount = passedInInputTensors.Length();
  std::vector<OrtValue *> inputValues;
  inputValues.reserve(passedInInputsCount);
  for (uint32_t i = 0; i < passedInInputsCount; i++) {
    OrtValue *value = JavascriptTensorToOrtValue(allocatorInfo, env, passedInInputTensors[i]);
    if (!value) {
      Napi::TypeError::New(env, "not a valid tensor input").ThrowAsJavaScriptException();
    }
    inputValues.push_back(value);
  }

  OrtAllocator *allocator;
  auto status = OrtCreateDefaultAllocator(&allocator);
  // TODO: scope guard for `allocator`

  size_t inputCount;
  status = OrtSessionGetInputCount(this->session_, &inputCount);
  size_t outputCount;
  status = OrtSessionGetOutputCount(this->session_, &outputCount);

  std::vector<const char *> inputNames;
  std::vector<const char *> outputNames;
  inputNames.reserve(inputCount);
  outputNames.reserve(outputCount);
  for (size_t i = 0; i < inputCount; i++) {
    char *str;
    OrtSessionGetInputName(this->session_, i, allocator, &str);
    inputNames.push_back(str);
  }
  for (size_t i = 0; i < outputCount; i++) {
    char *str;
    OrtSessionGetOutputName(this->session_, i, allocator, &str);
    outputNames.push_back(str);
  }

  std::vector<OrtValue *> outputValues(outputCount);

  status = OrtRun(this->session_, nullptr, &inputNames[0], &inputValues[0], inputCount, &outputNames[0], outputCount, &outputValues[0]);

  for (size_t i = 0; i < outputCount; i++) {
    Napi::Value tensorObject = OrtValueToJavascriptTensor(env, outputValues[i]);
  }

  return info.Env().Undefined();
}
