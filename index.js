const onnxjs = require('onnxjs');
onnxjs.InferenceSession = require('./lib/inference-session-override').OnnxRuntimeInferenceSession;

// Work items for future:
//
// 0 - fix memory leak for inference result (output OrtValue)
// 1 - support other tensor types (currently only support float32) - DONE
// 2 - typescript type declaration                                 - DONE
// 3 - integration with ONNX.js                                    - DONE
// 4 - refine API (a javascript wrapper on native module)          - DONE
// 5 - publish npm module
