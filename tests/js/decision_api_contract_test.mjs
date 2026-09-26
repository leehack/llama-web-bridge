// Static API contract checks for versioned Web decision-head support.
import { bridgeJsSource, methodBody, readRepoText } from './bridge_js_source.mjs';
import { readNativeCoreSource } from './native_core_source.mjs';
import { TEST_COMMAND } from '../../scripts/verify_ci_reliability.mjs';

const CORE = readNativeCoreSource();
const DECISION = readRepoText('src/llama_webgpu_decision.cpp');
const HEADER = readRepoText('src/llama_webgpu_decision.h');
const JS = bridgeJsSource();
const BRIDGE_JS = readRepoText('js/src/bridge.ts');
const DTS = readRepoText('js/src/llama_webgpu_bridge.d.ts');
const CMAKE = readRepoText('CMakeLists.txt');
const README = readRepoText('README.md');
const API_DOCS = readRepoText('docs/api.md');
// Python's " ".join(text.split()): collapse runs of Python str whitespace.
const API_DOCS_FLAT = API_DOCS
  .split(/[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/)
  .filter(Boolean)
  .join(' ');
const PACKAGE = readRepoText('package.json');
const SMOKE = readRepoText('scripts/decision_browser_smoke.mjs');
const CONTRACT_TEST = readRepoText('tests/js/decision_bridge_contract_test.mjs');

const NATIVE_EXPORTS = [
  'llamadart_webgpu_decision_api_version',
  'llamadart_webgpu_decision_capabilities_json',
  'llamadart_webgpu_decision_load',
  'llamadart_webgpu_decision_head_info_json',
  'llamadart_webgpu_decision_run',
  'llamadart_webgpu_decision_free',
];
const PUBLIC_METHODS = [
  'getDecisionCapabilities',
  'loadDecisionHead',
  'runDecision',
  'freeDecisionHead',
];

const errors = [];

function require(condition, message) {
  if (!condition) {
    errors.push(message);
  }
}

function includesAll(text, ...needles) {
  return needles.every((needle) => text.includes(needle));
}

function functionBody(source, signature) {
  const start = source.indexOf(signature);
  if (start < 0) {
    return '';
  }
  const end = source.indexOf('\n}\n', start);
  return source.slice(start, end > 0 ? end : source.length);
}

// Returns the worker host's dedicated handler for `method`, up to its return.
function workerHostBranch(method) {
  const host = functionBody(JS, 'function installBridgeWorkerHost() {');
  const start = host.indexOf(`if (method === '${method}') {`);
  if (start < 0) {
    return '';
  }
  const end = host.indexOf('return;', start);
  return host.slice(start, end > 0 ? end : host.length);
}

require(
  HEADER.includes('LLAMADART_WEBGPU_DECISION_API_VERSION = 1')
    && JS.includes('const DECISION_API_VERSION = 1;'),
  'decision API must have one explicit version shared by the core and the bridge',
);
for (const symbol of NATIVE_EXPORTS) {
  require(
    new RegExp(String.raw`EMSCRIPTEN_KEEPALIVE [^\n]*\b${symbol}\(`).test(CORE),
    `missing core wrapper ${symbol}`,
  );
  require(CMAKE.includes(`'_${symbol}'`), `missing exported symbol _${symbol}`);
}
require(
  CMAKE.includes('src/llama_webgpu_decision.cpp') && CMAKE.includes('"${LLAMA_CPP_DIR}/vendor"'),
  'CMake must build the decision module with the vendored nlohmann/json include path',
);

const freeRuntime = functionBody(CORE, 'void free_runtime() {');
require(
  freeRuntime.includes('free_decision_heads();')
    && freeRuntime.indexOf('free_decision_heads();') < freeRuntime.indexOf('llama_free(g_state.ctx)')
    && freeRuntime.indexOf('llama_free(g_state.ctx)') < freeRuntime.indexOf('llama_model_free(g_state.model)'),
  'decision heads must be freed before the shared context and model in free_runtime',
);
for (const signature of [
  'int32_t llamadart_webgpu_decision_load(',
  'int32_t llamadart_webgpu_decision_run(',
]) {
  require(
    functionBody(CORE, signature).includes('g_generation_active || g_tts_active'),
    `${signature.split('(')[0].split(/\s+/).at(-1)} must refuse to run during generation or text-to-speech`,
  );
}
require(
  CORE.includes('g_next_decision_handle++') && CORE.includes('g_next_decision_handle = 1')
    && !CORE.includes('g_next_decision_handle = 0'),
  'core decision handles must be monotonic within a runtime',
);

const run = functionBody(DECISION, 'llama_webgpu_decision_status llama_webgpu_decision_run(');
const validateAt = run.indexOf('llama_webgpu_decision_validate_sequences(');
require(
  validateAt >= 0 && validateAt < run.indexOf('llama_encode('),
  'every decision sequence must be validated before the first llama_encode',
);
require(
  DECISION.includes('token_count > static_cast<size_t>(std::max(token_limit, 0))')
    && DECISION.includes('llama_n_ubatch(head->context)'),
  "the sequence token limit must come from the head context's n_ubatch",
);
require(
  includesAll(
    DECISION,
    'context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;',
    'context_params.embeddings = true;',
    'context_params.n_seq_max = 1;',
  ),
  'each head needs a private per-token embedding context',
);
require(
  /backends\[backend_count\+\+\] = head->device_backend;.*?backends\[backend_count\+\+\] = head->cpu_backend;.*?ggml_backend_sched_new\(/s.test(DECISION),
  'the head scheduler must list the CPU backend last',
);
require(
  DECISION.includes('std::erf(') && DECISION.includes('ggml_gelu_erf('),
  'the head must use erf GELU on both the graph and the host act MLP',
);
require(
  includesAll(DECISION, 'kMaxHeaderBytes', 'static_assert(sizeof(off_t) == 8', 'fall outside the'),
  'safetensors parsing must bound the header and tensor offsets with 64-bit file offsets',
);
require(
  includesAll(
    DECISION,
    'ordered_json::parse(text, nullptr, false)',
    'ordered_json::parse(header_text, nullptr, false)',
    'error_handler_t::replace',
  ) && !DECISION.includes('throw '),
  'decision JSON handling must not depend on C++ exceptions',
);
for (const text of ['header_text', 'text']) {
  const parseAt = DECISION.indexOf(`ordered_json::parse(${text}, nullptr, false)`);
  const checkAt = DECISION.indexOf(`json_depth_within_limit(${text})`);
  require(
    checkAt >= 0 && checkAt < parseAt,
    `untrusted JSON (${text}) must pass the nesting limit before it is parsed`,
  );
}
const validate = functionBody(
  DECISION,
  'llama_webgpu_decision_status llama_webgpu_decision_validate_sequences(',
);
require(
  validate.includes('sequence.markers.size() > token_count'),
  "the marker count must be bounded by the sequence's token count",
);
require(
  HEADER.includes('const char * config_path;')
    && JS.includes('[headPath, label, configPath]')
    && /core\.FS\.writeFile\(configPath, textEncoder\.encode\(configJson!?\)\)/.test(JS),
  'configJson must reach the core as a WASMFS file, not a stack-copied ccall string',
);

for (const method of PUBLIC_METHODS) {
  require(
    (JS.match(new RegExp(String.raw`\basync\s+${method}\s*\(`, 'g')) ?? []).length >= 2
      || (method === 'getDecisionCapabilities' && JS.includes(`  ${method}() {`)),
    `expected direct-runtime and public ${method} methods`,
  );
  require(DTS.includes(method), `TypeScript declarations must expose ${method}`);
  require(API_DOCS.includes(method), `public API docs must document ${method}`);
}
for (const kind of [
  'decision-capabilities',
  'decision-head-load',
  'decision-run',
  'decision-head-free',
]) {
  require(JS.includes(`kind: '${kind}'`), `decision operation ${kind} must take the queue`);
}
const runBranch = workerHostBranch('runDecision');
require(
  runBranch.includes('transfers.push(buffer)')
    && runBranch.includes("self.postMessage({ type: 'result', id, value }, transfers)"),
  'worker decision runs must transfer output buffers instead of copying them',
);
require(
  workerHostBranch('loadDecisionHead').includes("event: 'progress'"),
  'worker head loads must post download progress so the request timer re-arms',
);
require(
  JS.includes('sequences * DECISION_WORKER_TIMEOUT_PER_SEQUENCE_MS'),
  'the worker runDecision timeout must grow with the number of sequences',
);
require(
  includesAll(
    JS,
    '_resolveDecisionHead(handle)',
    'entry.owner !== this._decisionOwner()',
    'or the bridge runtime restarted',
  ),
  'facade decision handles must be bound to the worker or runtime that loaded them',
);
require(
  /if \(!this\._shouldFallbackToMainThread\(error\)\) \{\s*throw error;/s.test(
    methodBody(BRIDGE_JS, 'async _loadDecisionHeadUnlocked('),
  ),
  'worker head-load errors must only fall back to the main thread for worker failures',
);
require(
  includesAll(
    JS,
    "_llamadart_webgpu_decision_capabilities_json !== 'function'",
    'this bridge needs version ${DECISION_API_VERSION}',
  ),
  'decision capabilities must probe the core export and API version',
);
require(
  includesAll(
    DTS,
    'interface DecisionCapabilities',
    'interface DecisionHeadInfo',
    'interface DecisionSequence',
    'interface DecisionOutput',
  ),
  'TypeScript declarations must expose decision capabilities, head info, sequences and outputs',
);
// npm test globs tests/**/*_test.mjs, so the contract test above runs once it
// exists under that name.
require(
  JSON.parse(PACKAGE).scripts?.test === TEST_COMMAND,
  'npm test must run the decision bridge contract test',
);
require(
  includesAll(
    CONTRACT_TEST,
    'the bridge runtime restarted',
    'malformed input never reaches the core',
    'an ordinary load error keeps the worker',
    'the model is reloaded on the main thread',
    'the load is retried on the main thread',
    'a failed worker free still falls back',
    'every output buffer is transferred',
    "_resolveRequestTimeoutMs('runDecision'",
    'decision API version 2',
  ),
  'decision contract test must cover stale handles, input validation, worker '
    + 'fallback scoping, worker loss, worker-host transfers, timeouts, and API version skew',
);
require(
  API_DOCS.includes('Decision heads')
    && API_DOCS_FLAT.includes('ModernBERT')
    && API_DOCS_FLAT.includes('not durable'),
  'public API docs must document decision heads, their encoder requirement, and handle lifetime',
);
require(
  README.includes('Decision heads') && README.includes('decision_browser_smoke.mjs'),
  'README must document decision heads and their real-model smoke',
);
require(
  includesAll(
    SMOKE,
    "RUNTIME_MODES = Object.freeze(['direct', 'worker'])",
    "flag: '--fixture-path'",
    "flag: '--config-path'",
    "flag: '--gpu-layers'",
    'checkParity(payload, args);',
    'worstLogitDiff',
    'argmaxChanges',
    'oversizedLength',
    'Invalid safetensors file',
    'freeDecisionHead',
  ),
  'real-model smoke must compare direct/worker outputs with the reference fixture, '
    + 'reject oversized sequences and invalid heads, and free heads',
);
const parityStart = SMOKE.indexOf('export function checkParity(');
const parity = parityStart >= 0
  ? SMOKE.slice(parityStart, SMOKE.indexOf('\n}\n', parityStart + 1))
  : '';
require(
  parity.includes("within('worstActRelativeDiff', args.maxActRelativeDiff)"),
  'real-model smoke must gate raw act logits, which saturated act probabilities cannot',
);
for (const stage of [
  'reject-deep-header',
  'reject-deep-config',
  'large-config',
  'reject-excess-markers',
]) {
  // The harness page is a template literal, so its own template's closing
  // backtick is escaped.
  require(SMOKE.includes(`:${stage}\\\``), `real-model smoke must run the ${stage} stage`);
}

if (errors.length > 0) {
  console.error('Decision API contract failed:');
  for (const item of errors) {
    console.error(`- ${item}`);
  }
  process.exit(1);
}

console.log('Decision API contract passed');
