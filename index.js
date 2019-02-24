module.exports = require('./bin/onnxruntime.node');

// Work items for future:
//
// 0 - fix memory leak for inference result (output OrtValue)
// 1 - support other tensor types (currently only support float32)
// 2 - typescript type declaration
// 3 - integration with ONNX.js
// 4 - refine API (a javascript wrapper on native module)
// 5 - publish npm module
