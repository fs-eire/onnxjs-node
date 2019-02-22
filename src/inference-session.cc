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

Napi::Value InferenceSession::Run(const Napi::CallbackInfo& info) {
  // TODO: implement session.run()
  return info.Env().Undefined();
}
