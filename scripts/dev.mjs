import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const root = fileURLToPath(new URL('../', import.meta.url));
const windows = process.platform === 'win32';
const python = path.join(root, 'backend', '.venv', windows ? 'Scripts/python.exe' : 'bin/python');
const frontend = path.join(root, 'frontend');
const cli = path.join(frontend, 'node_modules/vinext/dist/cli.js');
const children = new Set();
let stopping = false;

async function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  await Promise.all([...children].map(child => new Promise(resolve => {
    if (windows) {
      const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
      killer.once('error', resolve);
      killer.once('exit', resolve);
    } else {
      try { process.kill(-child.pid, 'SIGTERM'); } catch {}
      resolve();
    }
  })));
  process.exit(code);
}

function start(command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: 'inherit', detached: !windows });
  children.add(child);
  child.once('error', error => {
    children.delete(child);
    console.error(error.message);
    void stop(1);
  });
  child.once('exit', code => {
    children.delete(child);
    void stop(code ?? 1);
  });
  return child;
}

async function healthy() {
  try {
    const response = await fetch('http://127.0.0.1:8000/health', { signal: AbortSignal.timeout(1000) });
    const data = await response.json();
    return response.ok && data.status === 'ok' && data.milestone === 'noise-experiments';
  } catch { return false; }
}

process.on('SIGINT', () => void stop());
process.on('SIGTERM', () => void stop());

try {
  if (!existsSync(cli)) throw new Error('Frontend dependencies are missing. Run npm.cmd run setup first.');
  if (await healthy()) {
    console.log('Using the SpectraEdge backend already running on port 8000. It will remain running when this terminal stops.');
  } else {
    if (!existsSync(python)) throw new Error('Python environment is missing. Follow the install steps in README.md.');
    console.log('Starting Python backend at http://127.0.0.1:8000');
    start(python, ['-B', '-m', 'uvicorn', 'backend.app:app', '--reload', '--reload-dir', 'backend', '--host', '127.0.0.1', '--port', '8000'], root);
    let ready = false;
    for (let attempt = 0; attempt < 60 && !stopping; attempt++) {
      if (await healthy()) { ready = true; break; }
      await delay(500);
    }
    if (!ready) throw new Error('Python backend did not become ready. Check the error above and whether port 8000 is occupied.');
  }
  if (!stopping) {
    console.log('Backend ready. Starting website. Press Ctrl+C to stop services started here.');
    start(process.execPath, [cli, 'dev', ...process.argv.slice(2)], frontend);
  }
} catch (error) {
  console.error(error.message);
  await stop(1);
}
