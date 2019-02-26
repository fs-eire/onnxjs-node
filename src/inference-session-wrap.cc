#include <core/providers/cpu/cpu_provider_factory.h>

#include "inference-session-wrap.h"
#include "inference-session.h"
#include "utils.h"

Napi::FunctionReference InferenceSessionWrap::constructor;

Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "InferenceSession", {
    InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel),
    InstanceMethod("run", &InferenceSessionWrap::Run),
    InstanceAccessor("inputNames", &InferenceSessionWrap::GetInputNames, nullptr, napi_default, nullptr),
    InstanceAccessor("outputNames", &InferenceSessionWrap::GetOutputNames, nullptr, napi_default, nullptr)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("InferenceSession", func);
  return exports;
}

InferenceSessionWrap::InferenceSessionWrap(const Napi::CallbackInfo& info)
  : Napi::ObjectWrap<InferenceSessionWrap>(info)
  , initialized_(false)
  , session_(std::make_unique<InferenceSession>()) {}



Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (this->initialized_) {
    throw Napi::Error::New(env, "Model already loaded. Cannot load model multiple times.");
  }

  size_t length = info.Length();
  if (length <= 0 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Expect argument: model path");
  }

  Napi::String value = info[0].As<Napi::String>();

  try {
    this->session_->LoadModel(
#ifdef _WIN32
      reinterpret_cast<const wchar_t *>(value.Utf16Value().c_str())
#else
      value.Utf8Value().c_str()
#endif
    );
  } catch (std::exception const& e) {
    throw Napi::Error::New(env, e.what());
  }

  this->initialized_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::GetInputNames(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }
  return scope.Escape(CreateNapiArrayFrom(env, this->session_->GetInputNames()));
}

Napi::Value InferenceSessionWrap::GetOutputNames(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }
  return scope.Escape(CreateNapiArrayFrom(env, this->session_->GetOutputNames()));
}


Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);

  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }
  if (info.Length() <= 0) {
    throw Napi::TypeError::New(env, "Expect argument: input tensors");
  }
  if (!info[0].IsArray()) {
    throw Napi::TypeError::New(env, "Expect the first argument to be an array of input tensors");
  }
  auto inputTensors = info[0].As<Napi::Array>();
  auto inputTensorCount = inputTensors.Length();
  std::vector<Tensor> inputs;
  inputs.reserve(inputTensorCount);
  for (uint32_t i = 0; i < inputTensorCount; i++) {
    inputs.emplace_back(Tensor::From(inputTensors[i], this->session_->GetInputNames()[i].c_str()));
  }

  try {
      auto outputs = this->session_->Run(inputs);
      auto outputTensorCount = static_cast<uint32_t>(outputs.size());
      auto outputTensors = Napi::Array::New(env, outputTensorCount);
      for (uint32_t i = 0; i < outputTensorCount; i++) {
        outputTensors.Set(i, outputs[i].ToNapiValue(env));
        OrtReleaseValue(outputs[i].value);
      }

      return scope.Escape(outputTensors);
  } catch (std::exception const& e) {
      throw Napi::Error::New(env, e.what());
  }
}
