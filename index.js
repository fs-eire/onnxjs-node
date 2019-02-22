const fs = require('fs');
const path = require('path');

// create folder /bin
if (!fs.existsSync(path.join(__dirname, 'bin'))) {
    fs.mkdirSync(path.join(__dirname, 'bin'));
}

// copy file onnxruntime.node
fs.copyFileSync(path.join(__dirname, './build/Release/onnxruntime.node'), path.join(__dirname, 'bin', 'onnxruntime.node'));

// copy DLL
fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/mkldnn.dll'), path.join(__dirname, 'bin', 'mkldnn.dll'));
fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/onnxruntime.dll'), path.join(__dirname, 'bin', 'onnxruntime.dll'));

const ox = require('./bin/onnxruntime.node');

console.log(ox);

