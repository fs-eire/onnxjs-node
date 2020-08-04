import * as onnxjs from 'onnxjs';
import * as os from 'os';

// check if Node.js
if (typeof process !== 'undefined' && process && process.release && process.release.name === 'node') {
  // check if 64-bit platform
  if (os.arch() !== 'x64') {
    throw new Error(`onnxruntime does not support architecture '${os.arch()}'`);
  }

  // check if Linux or Windows
  if (['win32', 'linux', 'darwin'].indexOf(os.platform()) === -1) {
    throw new Error(`onnxruntime does not support platform '${os.platform()}'`);
  }

  // check endianness
  if (os.endianness() !== 'LE') {
    throw new Error(`onnxruntime node binding does not support non little-endian platform`);
  }

  // create a new onnx object and assign property 'InferenceSession'
  const onnx: typeof onnxjs = Object.create(onnxjs);
  Object.defineProperty(onnx, 'InferenceSession', {
    enumerable: true,
    get: function() {
      return require('./inference-session-override').OnnxRuntimeInferenceSession;
    }
  });

  (global as any).onnx = onnx;
}

export = onnx;
