const spawnSync = require('child_process').spawnSync;

const update = spawnSync('git submodule update --init --recursive', {shell: true, stdio: 'inherit', cwd: __dirname});
if (update.status !== 0) {
  if (update.error) {
    console.error(update.error);
  }
  process.exit(update.status);
}