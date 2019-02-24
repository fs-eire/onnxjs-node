#include <napi.h>
#include "inference-session-wrap.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  return InferenceSessionWrap::Init(env, exports);
}

NODE_API_MODULE(onnxruntime, InitAll)
