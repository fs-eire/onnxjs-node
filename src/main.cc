#include <napi.h>
#include "inference-session-wrap.h"
#include "inference-session.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InferenceSession::Init();
  return InferenceSessionWrap::Init(env, exports);
}

NODE_API_MODULE(onnxruntime, InitAll)
