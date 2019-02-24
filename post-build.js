const fs = require('fs');
const path = require('path');
const os = require('os');

// command line flags
const debug = process.argv.slice(2).indexOf('--debug') !== -1 || process.argv.slice(2).indexOf('-d') !== -1;

// create folder /bin
if (!fs.existsSync(path.join(__dirname, 'bin'))) {
    fs.mkdirSync(path.join(__dirname, 'bin'));
}

// configs
const BUILD_TYPE = debug ? 'Debug' : 'Release';

if (os.platform() === 'win32') {
    // copy file onnxruntime.node
    fs.copyFileSync(path.join(__dirname, `./build/${BUILD_TYPE}/onnxruntime.node`), path.join(__dirname, 'bin', 'onnxruntime.node'));
    if (fs.existsSync(path.join(__dirname, `./build/${BUILD_TYPE}/onnxruntime.pdb`))) {
        fs.copyFileSync(path.join(__dirname, `./build/${BUILD_TYPE}/onnxruntime.pdb`), path.join(__dirname, 'bin', 'onnxruntime.pdb'));
    }

    // copy DLL
    fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/mkldnn.dll'), path.join(__dirname, 'bin', 'mkldnn.dll'));
    fs.copyFileSync(path.join(__dirname, './onnxruntime/bin/win-x64/native/onnxruntime.dll'), path.join(__dirname, 'bin', 'onnxruntime.dll'));

} else if (os.platform() === 'darwin') {
    // TODO: add support to macOS
    throw new Error('currently not support macOS');
} else {
    // linux

    // TODO: copy linux binaries
}
