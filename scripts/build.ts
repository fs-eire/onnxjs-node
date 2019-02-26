import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as rimraf from 'rimraf';

// command line flags
const DEBUG = process.argv.slice(2).indexOf('--debug') !== -1 || process.argv.slice(2).indexOf('-d') !== -1;

// configs
const BUILD_TYPE = DEBUG ? 'Debug' : 'Release';

// build paths
const ROOT = path.join(__dirname, '..');
const BIN = path.join(ROOT, 'bin');
const NPM_BIN_FOLDER = execSync('npm bin', {encoding: 'utf8'}).trimRight();
const CMAKE_JS_FULL_PATH = path.join(NPM_BIN_FOLDER, 'cmake-js');
const BUILD_OUTPUT_FOLDER = path.join(ROOT, 'build', BUILD_TYPE);
const ONNXRUNTIME_DIST = path.join(ROOT, 'onnxruntime');

// ====================

console.log('BUILD [1/3] syncing submodules ...');

const update = spawnSync('git submodule update --init --recursive', {shell: true, stdio: 'inherit', cwd: ROOT});
if (update.status !== 0) {
  if (update.error) {
    console.error(update.error);
  }
  process.exit(update.status);
}

console.log('BUILD [2/3] build node binding ...');

const cmakejsArgs = ['compile', '-G"Visual Studio 15 2017 Win64"'];
if (DEBUG) cmakejsArgs.push('-D');

const cmakejs = spawnSync(CMAKE_JS_FULL_PATH, cmakejsArgs, {shell: true, stdio: 'inherit'});
if (cmakejs.status !== 0) {
  if (cmakejs.error) {
    console.error(cmakejs.error);
  }
  process.exit(cmakejs.status);
}

console.log('BUILD [3/3] binplace build artifacts ...');

if (fs.existsSync(BIN)) {
  rimraf.sync(BIN);
}
fs.mkdirSync(BIN);

if (os.platform() === 'win32') {
  // copy file onnxruntime.node
  fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.node'), path.join(BIN, 'onnxruntime.node'));

  // copy pdb file (this is useful in DEBUG mode)
  if (DEBUG && fs.existsSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.pdb'))) {
    fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.pdb'), path.join(BIN, 'onnxruntime.pdb'));
  }

  // copy DLL
  fs.copyFileSync(path.join(ONNXRUNTIME_DIST, 'bin/win-x64/native/mkldnn.dll'), path.join(BIN, 'mkldnn.dll'));
  fs.copyFileSync(path.join(ONNXRUNTIME_DIST, 'bin/win-x64/native/onnxruntime.dll'), path.join(BIN, 'onnxruntime.dll'));
} else if (os.platform() === 'darwin') {
  throw new Error('currently not support macOS');
} else {
  // linux

  // TODO: copy linux binaries
}
