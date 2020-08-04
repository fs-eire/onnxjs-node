// import binding
import '..';

import * as assert from 'assert';
import * as fs from 'fs';
import * as onnx_proto from 'onnx-proto';
import {InferenceSession, Tensor} from 'onnxjs';
import * as path from 'path';

// ONNXRuntime does not support non 4-dimensional inputs for ConvTranspose
// https://github.com/Microsoft/onnxruntime/blob/v0.1.5/onnxruntime/core/providers/cpu/nn/conv_transpose.cc#L89-L93
const SKIPPED_TESTS = [
  'test_convtranspose_1d',
  'test_convtranspose_3d',
];

const NODE_TESTS_ROOT = path.join(__dirname, '../deps/onnx/onnx/backend/test/data/node');
const models = fs.readdirSync(NODE_TESTS_ROOT);
for (const model of models) {
  // skipped tests
  if (SKIPPED_TESTS.indexOf(model) !== -1) {
    continue;
  }

  // read each model folders
  const modelFolder = path.join(NODE_TESTS_ROOT, model);
  let modelPath: string;
  const modelTestCases: Array<[(Tensor | undefined)[], (Tensor | undefined)[]]> = [];
  for (const currentFile of fs.readdirSync(modelFolder)) {
    const currentPath = path.join(modelFolder, currentFile);
    const stat = fs.lstatSync(currentPath);
    if (stat.isFile()) {
      const ext = path.extname(currentPath);
      if (ext.toLowerCase() === '.onnx') {
        modelPath = currentPath;
      }
    } else if (stat.isDirectory()) {
      const inputs: (Tensor|undefined)[] = [];
      const outputs: (Tensor|undefined)[] = [];
      for (const dataFile of fs.readdirSync(currentPath)) {
        const dataFileFullPath = path.join(currentPath, dataFile);
        const ext = path.extname(dataFile);

        if (ext.toLowerCase() === '.pb') {
          let tensor: Tensor|undefined = undefined;
          try {
            tensor = loadTensorFromFile(dataFileFullPath);
          } catch (e) {
            // skip
          }

          if (dataFile.indexOf('input') !== -1) {
            inputs.push(tensor);
          } else if (dataFile.indexOf('output') !== -1) {
            outputs.push(tensor);
          }
        }
      }
      modelTestCases.push([inputs, outputs]);
    }
  }

  // add cases
  describe(`node - ${model}`, () => {
    let session: InferenceSession;
    before(async () => {
      session = new onnx.InferenceSession({backendHint: 'onnxruntime'});
      await session.loadModel(modelPath);
    });

    for (let i = 0; i < modelTestCases.length; i++) {
      const testCase = modelTestCases[i];
      const inputs = testCase[0];
      const expectedOutputs = testCase[1];
      const skip = inputs.some(t => t === undefined) || expectedOutputs.some(t => t === undefined);
      (skip ? it.skip : it)(`case${i}`, async () => {
        const outputs = await session.run(inputs.map(t => t!));

        let j = 0;
        outputs.forEach((output) => {
          assert.ok(areEqual(output, expectedOutputs[j]!));
          j++;
        });
      });
    }
  });
}

function loadTensorFromFile(pbFile: string): Tensor {
  const tensorProto = onnx_proto.onnx.TensorProto.decode(fs.readFileSync(pbFile));
  let transferredTypedArray: Uint8Array|Float32Array|Int32Array;
  let type: Tensor.Type;
  switch (tensorProto.dataType) {
    case onnx_proto.onnx.TensorProto.DataType.FLOAT:
      transferredTypedArray = new Float32Array(tensorProto.rawData.byteLength / 4);
      type = 'float32';
      break;
    case onnx_proto.onnx.TensorProto.DataType.INT32:
      transferredTypedArray = new Int32Array(tensorProto.rawData.byteLength / 4);
      type = 'int32';
      break;
    case onnx_proto.onnx.TensorProto.DataType.BOOL:
      transferredTypedArray = new Uint8Array(tensorProto.rawData.byteLength);
      type = 'bool';
      break;
    default:
      throw new Error(`not supported tensor type: ${tensorProto.dataType}`);
  }
  const transferredTypedArrayRawDataView =
      new Uint8Array(transferredTypedArray.buffer, transferredTypedArray.byteOffset, tensorProto.rawData.byteLength);
  transferredTypedArrayRawDataView.set(tensorProto.rawData);

  const dims = tensorProto.dims.map((dim) => typeof dim === 'number' ? dim : dim.toNumber());

  return new Tensor(transferredTypedArray, type, dims);
}



// This function check whether 2 tensors should be considered as 'match' or not
function areEqual(actual: Tensor, expected: Tensor): boolean {
  if (!actual || !expected) {
    return false;
  }
  if (!actual.dims || !expected.dims) {
    return false;
  }

  const actualDims = actual.dims;
  const actualType = actual.type;
  const expectedDims = expected.dims;
  const expectedType = expected.type;

  if (actualType !== expectedType) {
    return false;
  }
  if (actualDims.length !== expectedDims.length) {
    return false;
  }

  for (let i = 0; i < actualDims.length; i++) {
    if (actualDims[i] !== expectedDims[i]) {
      return false;
    }
  }

  switch (actualType) {
    case 'float32':
      return floatEqual(
          actual.data as number[] | Float32Array | Float64Array,
          expected.data as number[] | Float32Array | Float64Array);

    case 'int32':
    case 'bool':
      return integerEqual(
          actual.data as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array,
          expected.data as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array);

    case 'string':
    default:
      throw new Error('type not implemented or not supported');
  }
}

function floatEqual(actual: number[]|Float32Array|Float64Array, expected: number[]|Float32Array|Float64Array): boolean {
  const THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
  const THRESHOLD_RELATIVE_ERROR = 1.000001;
  if (actual.length !== expected.length) {
    return false;
  }

  for (let i = actual.length - 1; i >= 0; i--) {
    const a = actual[i], b = expected[i];

    // check for NaN
    //
    if (Number.isNaN(a) && Number.isNaN(b)) {
      continue;  // 2 numbers are NaN, treat as equal
    }
    if (Number.isNaN(a) || Number.isNaN(b)) {
      return false;  // one is NaN and the other is not
    }

    // Comparing 2 float numbers: (Suppose a >= b)
    //
    // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
    //   test pass
    // else
    //   test fail
    // endif
    //
    if (Math.abs(actual[i] - expected[i]) < THRESHOLD_ABSOLUTE_ERROR) {
      continue;  // absolute error check pass
    }
    if (a !== 0 && b !== 0 && a * b > 0 && a / b < THRESHOLD_RELATIVE_ERROR && b / a < THRESHOLD_RELATIVE_ERROR) {
      continue;  // relative error check pass
    }

    // if code goes here, it means both (abs/rel) check failed.
    return false;
  }

  return true;
}

function integerEqual(
    actual: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array,
    expected: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array): boolean {
  if (actual.length !== expected.length) {
    return false;
  }

  for (let i = actual.length - 1; i >= 0; i--) {
    if (actual[i] !== expected[i]) {
      return false;
    }
  }

  return true;
}