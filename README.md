[![npm version](https://badge.fury.io/js/onnxjs-node.svg)](https://badge.fury.io/js/onnxjs-node)
[![GitHub version](https://badge.fury.io/gh/fs-eire%2Fonnxjs-node.svg)](https://badge.fury.io/gh/fs-eire%2Fonnxjs-node)

# onnxjs-node
**onnxjs-node** is a Node.js binding of [ONNXRuntime](https://github.com/Microsoft/onnxruntime) that works seamlessly with [ONNX.js](https://github.com/Microsoft/onnxjs).

## Installation
Install the latest stable version:
```
npm install onnxjs-node
```

*NOTE: binary files will be pulled from github during the `npm install` process.*

## Supported Platforms
OS       |Arch |CPU/GPU |NAPI version |Node.js version | ONNXRuntime version
---------|-----|--------|-------------|----------------|---------------------
 Windows | x64 | CPU    | v3          | v12.0.0+       | v1.4.0
 Linux   | x64 | CPU    | v3          | v12.0.0+       | v1.4.0
 macOS   | x64 | CPU    | v3          | v12.0.0+       | v1.4.0
 Windows | x64 | GPU    | v3          | v12.0.0+       | v1.4.0
 Linux   | x64 | GPU    | v3          | v12.0.0+       | v1.4.0

## Usage
There are 2 options to import `onnxjs-node`.
 -  Option 1 - replace `onnxjs` by `onnxjs-node`:
    ```js
    //const onnx = require('onnxjs');
    const onnx = require('onnxjs-node');

    // use 'onnx'
    // ...
    ```
 -  Option 2 - add a single line to require `onnxjs-node`:
    ```js
    const onnx = require('onnxjs');
    require('onnxjs-node');  // this line can be put on the top as well

    // use 'onnx'
    // ...
    ```

After `onnxjs-node` is imported, the default inference session class of ONNX.js will be overwritten. Any existing ONNX.js code will continue to work and model will run by ONNXRuntime backend.

## Options
### Enable/Disable GPU
Coming soon...
### Backend Fallback
After `onnxjs-node` is imported, ONNXRuntime backend will be used by default. However, it is possible to fallback to other backend by specifying the session option `backendHint`:
```js
session = new onnx.InferenceSession({backendHint: 'wasm'});  // use WebAssembly backend
```

## Documentation
- [ONNX.js Home](https://github.com/Microsoft/onnxjs)
- [ONNXRuntime Node.js binding](https://github.com/Microsoft/onnxruntime/nodejs)

# License
Copyright (c) fs-eire. All rights reserved.

Licensed under the [MIT](https://github.com/fs-eire/onnxjs-node/blob/master/LICENSE) License.
