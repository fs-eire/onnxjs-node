#ifndef ONNXRUNTIME_NODE_INFERENCE_SESSION_WRAP_H
#define ONNXRUNTIME_NODE_INFERENCE_SESSION_WRAP_H

#pragma once

#include <memory>
#include <napi.h>

class InferenceSession;

class InferenceSessionWrap : public Napi::ObjectWrap<InferenceSessionWrap> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  InferenceSessionWrap(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;

  Napi::Value LoadModel(const Napi::CallbackInfo &info);

  // following functions have to be called after model is loaded.
  Napi::Value GetInputNames(const Napi::CallbackInfo &info);
  Napi::Value GetOutputNames(const Napi::CallbackInfo &info);
  Napi::Value Run(const Napi::CallbackInfo &info);

  bool initialized_;
  std::unique_ptr<InferenceSession> session_;
};

#endif
