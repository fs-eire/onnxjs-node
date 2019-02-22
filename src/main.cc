#include <napi.h>
#include "inference-session.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  return InferenceSession::Init(env, exports);
}

NODE_API_MODULE(onnxruntime, InitAll)
