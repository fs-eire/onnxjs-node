const os = require('os');

const onnxjs = require('onnxjs');

// check if Node.js
if (typeof process !== 'undefined' && process && process.release && process.release.name === 'node') {
    // check if 64-bit platform
    if (os.arch() !== 'x64') {
        throw new Error(`onnxruntime does not support architecture '${os.arch()}'`);
    }

    // check if Linux or Windows
    if (['win32', 'linux'].indexOf(os.platform()) === -1) {
        throw new Error(`onnxruntime does not support platform '${os.platform()}'`);
    }

    // check endianness
    if (os.endianness() !== 'LE') {
        throw new Error(`onnxruntime node binding does not support non little-endian platform`);
    }

    onnxjs.InferenceSession = require('./lib/inference-session-override').OnnxRuntimeInferenceSession;
}

// Work items for future:
//
// 0 - fix memory leak for inference result (output OrtValue)      - DONE
// 1 - support other tensor types (currently only support float32) - DONE
// 2 - typescript type declaration                                 - DONE
// 3 - integration with ONNX.js                                    - DONE
// 4 - refine API (a javascript wrapper on native module)          - DONE
// 5 - publish npm module
