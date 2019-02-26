#include "inference-session-wrap.h"
#include "inference-session.h"
#include <napi.h>

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InferenceSession::Init();
  return InferenceSessionWrap::Init(env, exports);
}

NODE_API_MODULE(onnxruntime, InitAll)
