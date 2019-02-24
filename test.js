const fs = require('fs');
const path = require('path');
const onnx_proto = require('onnx-proto');
const assert = require('assert');

const onnxruntime = require('.');

const sess = new onnxruntime.InferenceSession();
sess.loadModel(path.join(__dirname, 'data/data/test/onnx/v7/resnet50/model.onnx'));

const inputTensor0 = onnx_proto.onnx.TensorProto.decode(fs.readFileSync(path.join(__dirname, 'data/data/test/onnx/v7/resnet50/test_data_set_0/input_0.pb')));
const outputTensor0 = onnx_proto.onnx.TensorProto.decode(fs.readFileSync(path.join(__dirname, 'data/data/test/onnx/v7/resnet50/test_data_set_0/output_0.pb')));
const rawDataInput0 = new Float32Array(1*3*224*224);
const rawDataInput0Uint8 = new Uint8Array(rawDataInput0.buffer, rawDataInput0.byteOffset, rawDataInput0.length * 4);
rawDataInput0Uint8.set(inputTensor0.rawData);

const rawDataOutput0 = new Float32Array(1*1000);
const rawDataOutput0Uint8 = new Uint8Array(rawDataOutput0.buffer, rawDataOutput0.byteOffset, rawDataOutput0.length * 4);
rawDataOutput0Uint8.set(outputTensor0.rawData);

const result = sess.run([{dims: [1,3,224,224], data: rawDataInput0}]);

assert(Array.isArray(result));
assert(result.length === 1);
assert(result[0]);
assert(floatEqual(rawDataOutput0, result[0].data));


function floatEqual(actual, expected) {
    var THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
    var THRESHOLD_RELATIVE_ERROR = 1.000001;
    
    if (actual.length !== expected.length) {
        return false;
    }
    for (var i = actual.length - 1; i >= 0; i--) {
        var a = actual[i], b = expected[i];
        // check for NaN
        //
        if (Number.isNaN(a) && Number.isNaN(b)) {
            continue; // 2 numbers are NaN, treat as equal
        }
        if (Number.isNaN(a) || Number.isNaN(b)) {
            return false; // one is NaN and the other is not
        }
        // sign should be same if not equals to zero
        //
        if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
            return false; // sign is different
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
            continue; // absolute error check pass
        }
        if (a !== 0 && b !== 0 && a / b < THRESHOLD_RELATIVE_ERROR && b / a < THRESHOLD_RELATIVE_ERROR) {
            continue; // relative error check pass
        }
        // if code goes here, it means both (abs/rel) check failed.
        return false;
    }
    return true;
};