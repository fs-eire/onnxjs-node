const fs = require('fs');
const path = require('path');
const onnx_proto = require('onnx-proto');

// create folder /bin
if (!fs.existsSync(path.join(__dirname, 'bin'))) {
    fs.mkdirSync(path.join(__dirname, 'bin'));
}

// copy file onnxruntime.node
fs.copyFileSync(path.join(__dirname, './build/Debug/onnxruntime.node'), path.join(__dirname, 'bin', 'onnxruntime.node'));
fs.copyFileSync(path.join(__dirname, './build/Debug/onnxruntime.pdb'), path.join(__dirname, 'bin', 'onnxruntime.pdb'));

// copy DLL
fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/mkldnn.dll'), path.join(__dirname, 'bin', 'mkldnn.dll'));
fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/onnxruntime.dll'), path.join(__dirname, 'bin', 'onnxruntime.dll'));

const ox = require('./bin/onnxruntime.node');

const sess = new ox.InferenceSession();
sess.loadModel('D:\\eire\\onnxjs\\deps\\data\\data\\test\\onnx\\v7\\resnet50\\model.onnx');

const inputTensor0 = onnx_proto.onnx.TensorProto.decode(fs.readFileSync('D:\\eire\\onnxjs\\deps\\data\\data\\test\\onnx\\v7\\resnet50\\test_data_set_0\\input_0.pb'));
const rawData = new Float32Array(1*3*224*224);
const rawDataUint8 = new Uint8Array(rawData.buffer, rawData.byteOffset, rawData.length * 4);
rawDataUint8.set(inputTensor0.rawData);

const result = sess.run([{dims: [1,3,224,224], data: rawData}]);
