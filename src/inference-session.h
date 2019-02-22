#ifndef INFERENCE_SESSION_H
#define INFERENCE_SESSION_H

#include <napi.h>

#include "core/session/onnxruntime_c_api.h"

class InferenceSession : public Napi::ObjectWrap<InferenceSession> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  InferenceSession(const Napi::CallbackInfo& info);

 private:
  static Napi::FunctionReference constructor;

  Napi::Value LoadModel(const Napi::CallbackInfo& info);
  Napi::Value Run(const Napi::CallbackInfo& info);

  OrtEnv *env_;
  OrtSessionOptions *sessionOptions_;
  OrtSession *session_;
};

#endif
