// Live GitHub transport for the stable release orchestrator, the Node port of
// scripts/release_orchestrator_transport.py.
//
// The Gateway protocol is what every live proof and dispatch is written
// against; GhGateway implements it with the gh CLI, keeping read traffic and
// dispatch on distinct tokens. Every call is synchronous, as in Python.
//
// Gateway (duck-typed; tests/release/orchestrator/fixtures.mjs has a fake):
//   apiJson(path, { paginate = false, privileged = false } = {}) -> JSON value
//   downloadBytes(path, { accept, privileged = false }) -> Buffer
//   dispatchIdentity() -> string | null
//   releaseAttestation({ repository, releaseTag }) -> JSON value
//   dispatchWorkflow({ workflowFile, ref, inputs }) -> undefined
//   sleep(seconds) -> undefined
//   utcNow() -> "YYYY-MM-DDTHH:MM:SSZ"

import { spawnSync } from 'node:child_process';
import process from 'node:process';

import { BRIDGE_REPOSITORY, ContractError, strictJsonLoads } from '../contract.mjs';
import {
  PyException, isDict, pyDecodeUtf8, pyGet, pyJsonDumps, pyStrftimeUtc, pyStrip,
} from '../json.mjs';
import { isOSError, osErrorString } from '../python_compat.mjs';

const GH_MAX_BUFFER = 1024 * 1024 * 1024;

// bytes.decode("utf-8", "replace").
function decodeReplace(bytes) {
  return new TextDecoder('utf-8', { ignoreBOM: true }).decode(bytes);
}

// The OSError subprocess.run raises when the executable cannot be started.
function spawnError(error, executable) {
  if (!isOSError(error)) return error;
  const classes = { ENOENT: 'FileNotFoundError', EACCES: 'PermissionError', EPERM: 'PermissionError' };
  const exception = new PyException(classes[error.code] ?? 'OSError', osErrorString(error, executable));
  exception.code = error.code;
  return exception;
}

// time.sleep(seconds), blocking the thread as Python does.
function sleepSeconds(seconds) {
  if (!(seconds >= 0)) throw new PyException('ValueError', 'sleep length must be non-negative');
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, seconds * 1000);
}

// datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ").
export function utcNowText(now = new Date()) {
  return pyStrftimeUtc({
    year: now.getUTCFullYear(),
    month: now.getUTCMonth() + 1,
    day: now.getUTCDate(),
    hour: now.getUTCHours(),
    minute: now.getUTCMinutes(),
    second: now.getUTCSeconds(),
  });
}

// gh-backed transport. Read traffic and dispatch use distinct tokens.
//
// `gh` (the executable), `env` (the environment the child inherits, read at
// call time; process.env by default), `clock` (a function returning the
// current Date) and `sleeper` are injectable for tests.
export class GhGateway {
  constructor({
    readToken, dispatchToken, gh = 'gh', env = null, clock = () => new Date(), sleeper = sleepSeconds,
  }) {
    this.readToken = readToken;
    this.dispatchToken = dispatchToken || null;
    this.gh = gh;
    this.env = env;
    this.clock = clock;
    this.sleeper = sleeper;
  }

  // The child environment: a copy of the environment with GH_TOKEN set to
  // `token` and GITHUB_TOKEN removed.
  childEnv(token) {
    const env = { ...(this.env ?? process.env) };
    env.GH_TOKEN = token;
    delete env.GITHUB_TOKEN;
    return env;
  }

  // _run(args, privileged=..., binary=...): args[0] is "gh". Returns stdout.
  run(args, { privileged }) {
    const token = privileged ? this.dispatchToken : this.readToken;
    if (!token) throw new ContractError('no credential is available for this GitHub request; refusing to continue');
    const executable = args[0] === 'gh' ? this.gh : args[0];
    const completed = spawnSync(executable, args.slice(1), {
      env: this.childEnv(token),
      stdio: ['inherit', 'pipe', 'pipe'],
      maxBuffer: GH_MAX_BUFFER,
    });
    if (completed.error) throw spawnError(completed.error, args[0]);
    if (completed.status !== 0) {
      const stderr = pyStrip(decodeReplace(completed.stderr));
      throw new ContractError(`gh ${args.slice(1).join(' ')} failed: ${stderr}`);
    }
    return completed.stdout;
  }

  apiJson(path, { paginate = false, privileged = false } = {}) {
    const args = ['gh', 'api'];
    if (paginate) args.push('--paginate', '--slurp');
    args.push(path);
    const stdout = this.run(args, { privileged });
    return strictJsonLoads(pyDecodeUtf8(stdout), `gh api ${path}`);
  }

  downloadBytes(path, { accept, privileged = false }) {
    return this.run(['gh', 'api', '-H', `Accept: ${accept}`, path], { privileged });
  }

  dispatchIdentity() {
    if (!this.dispatchToken) return null;
    let payload;
    try {
      payload = this.apiJson('user', { privileged: true });
    } catch (error) {
      if (error instanceof ContractError) return null;
      throw error;
    }
    const login = isDict(payload) ? pyGet(payload, 'login') : null;
    return typeof login === 'string' && login ? login : null;
  }

  releaseAttestation({ repository, releaseTag }) {
    const stdout = this.run(['gh', 'release', 'verify', releaseTag, '--repo', repository, '--format', 'json'], { privileged: false });
    return strictJsonLoads(pyDecodeUtf8(stdout), `gh release verify ${repository}@${releaseTag}`);
  }

  // The inputs go to `gh workflow run --json` on stdin as
  // json.dumps(dict(inputs), sort_keys=True).
  dispatchWorkflow({ workflowFile, ref, inputs }) {
    if (!this.dispatchToken) throw new ContractError('workflow dispatch requires an orchestrator credential');
    const completed = spawnSync(this.gh, ['workflow', 'run', workflowFile, '--repo', BRIDGE_REPOSITORY, '--ref', ref, '--json'], {
      input: Buffer.from(pyJsonDumps({ ...inputs }, { sortKeys: true }), 'utf8'),
      env: this.childEnv(this.dispatchToken),
      stdio: ['pipe', 'pipe', 'pipe'],
      maxBuffer: GH_MAX_BUFFER,
    });
    if (completed.error) throw spawnError(completed.error, 'gh');
    if (completed.status !== 0) {
      const stderr = pyStrip(decodeReplace(completed.stderr));
      throw new ContractError(`dispatching ${workflowFile} failed: ${stderr}`);
    }
  }

  sleep(seconds) {
    this.sleeper(seconds);
  }

  utcNow() {
    return utcNowText(this.clock());
  }
}
