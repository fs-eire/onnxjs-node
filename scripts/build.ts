import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as rimraf from 'rimraf';
import * as fs_extra from 'fs-extra';

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
const ONNXRUNTIME_DIST = path.join(ROOT, 'onnxruntime', 'bin');

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
  fs.mkdirSync(path.join(BIN, 'win-x64'));
  fs_extra.copySync(path.join(ONNXRUNTIME_DIST, 'win-x64'), path.join(BIN, 'win-x64'));

  fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.node'), path.join(BIN, 'win-x64', 'onnxruntime.node'));
  if (DEBUG && fs.existsSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.pdb'))) {
    fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime.pdb'), path.join(BIN, 'win-x64', 'onnxruntime.pdb'));
  }

  fs.mkdirSync(path.join(BIN, 'win_gpu-x64'));
  fs_extra.copySync(path.join(ONNXRUNTIME_DIST, 'win_gpu-x64'), path.join(BIN, 'win_gpu-x64'));

  fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime_gpu.node'), path.join(BIN, 'win_gpu-x64', 'onnxruntime_gpu.node'));
  if (DEBUG && fs.existsSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime_gpu.pdb'))) {
    fs.copyFileSync(path.join(BUILD_OUTPUT_FOLDER, 'onnxruntime_gpu.pdb'), path.join(BIN, 'win_gpu-x64', 'onnxruntime_gpu.pdb'));
  }
} else if (os.platform() === 'darwin') {
  throw new Error('currently not support macOS');
} else {
  // linux

  // TODO: copy linux binaries
}
