#!/usr/bin/env node
// Artifact-driven automatic stable Web bridge release state machine: the CLI
// entry that .github/workflows/auto_llama_cpp_update.yml runs. It replaced
// the Python scripts/stable_release_orchestrator.py.
//
// Each event-driven scan resolves every stable native release after the
// immutable automation baseline, then idempotently advances each three-stage
// pipeline. A native release older than one a published asset release
// already records is superseded, not rebuilt. A daily scheduled scan provides
// repair fallback:
//
// 1. Build Exact Bridge Candidate      (.github/workflows/bridge_candidate.yml)
// 2. Qualify Exact Bridge Candidate    (.github/workflows/bridge_qualification.yml)
// 3. Publish Exact Qualified Assets    (.github/workflows/publish_assets.yml)
//
// Every transition is proven from downloaded artifact and release bytes. The
// live actions/runs API never echoes a run's dispatch inputs, so pipeline
// state is carried by a deterministic run-name that each workflow renders
// from its own exact inputs, and every named run is then re-proven against its
// run record, its unique artifact, and that artifact's contents before it
// advances anything.
//
// The state machine lives in the sibling modules; this file keeps the
// governed-path classifier (the source of truth that AGENTS.md and
// CONTRIBUTING.md name), the bridge source identity it resolves, and the
// command-line surface. A new sibling module must be listed in
// ORCHESTRATION_ONLY_PATHS below and in TOOLING in scripts/ci_scope.mjs.
//
// main(argv, { env, createGateway, stdout, stderr }) is the entry the tests
// drive: `env` replaces process.env, createGateway({ readToken,
// dispatchToken, env }) replaces the GhGateway constructor, and stdout and
// stderr receive text. It returns the exit status; a ContractError, an
// argparse exit or any other Python exception escapes to the caller, as it
// escapes Python's main() (runMain reports them the way the Python script's
// `__main__` block and interpreter do).

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import process from 'node:process';

import { ArgparseExit, parseArguments, progName } from '../cli.mjs';
import { ContractError, strictJsonLoads } from '../contract.mjs';
import {
  PyException, compareCodePoints, isDict, pyDecodeUtf8, pyGet, pyIsDir, pyIsSymlink, pyJoinPath, pyJsonDumps, pyKeys,
  pyOSError, pyPath, pyReadText, pyRepr, pySplitlines, pyStr, pyStrip, pyUniversalNewlines,
} from '../json.mjs';
import { isOSError, osErrorString, pyCompareIntTuples, pyFullmatch } from '../python_compat.mjs';
import { advancePipeline } from './driver.mjs';
import {
  COMMIT_RE, NativeProvenance, OrchestrationAction, REPOSITORY_OWNER, requireStableProvenance,
} from './model.mjs';
import {
  CHANNELS, nativeReleaseOrder, scanNativeProvenance, selectStableNativeBacklog,
} from './native.mjs';
import { computeCorrelationId } from './run_names.mjs';
import { makeDirectories } from './stage_proofs.mjs';
import { GhGateway } from './transport.mjs';

// These files can change how a release is discovered, qualified, or
// published, but they do not change the runtime/build inputs placed in the
// bridge artifact. Everything not explicitly classified here is governed by
// default so a newly added build input cannot silently inherit an older
// release identity. This is the source of truth for the classifier. A path
// stays listed after its file is deleted: history must not classify the
// deleting commit as a build input.
export const ORCHESTRATION_ONLY_PATHS = Object.freeze(new Set([
  '.gitignore',
  '.github/workflows/auto_llama_cpp_update.yml',
  '.github/workflows/bridge_qualification.yml',
  '.github/workflows/ci.yml',
  '.github/workflows/publish_assets.yml',
  'AGENTS.md',
  'CONTRIBUTING.md',
  'LICENSE',
  'README.md',
  // Deleted with the Node port of verify_ci_reliability, which was their
  // last importer.
  'scripts/bridge_js_source.py',
  'scripts/bridge_operation_queue_direct_cases.mjs',
  'scripts/bridge_operation_queue_fixtures.mjs',
  'scripts/bridge_operation_queue_lifecycle_contract_cases.mjs',
  'scripts/bridge_operation_queue_worker_proxy_cases.mjs',
  // Shared by the scripts/*_browser_smoke.mjs harnesses, which the
  // _browser_smoke.mjs suffix covers.
  'scripts/browser_smoke_support.mjs',
  // Ported to scripts/ci_scope.mjs; the .py path was deleted.
  'scripts/ci_scope.mjs',
  'scripts/ci_scope.py',
  // Deleted with the Node port of mtmd_compat_contract_test.py, its last
  // importer; tests/js/native_core_source.mjs is its twin.
  'scripts/native_core_source.py',
  // Deleted with the Node port of verify_ci_reliability; see above.
  'scripts/orchestrator_source.py',
  // The Python release orchestrator, deleted when the scan workflow switched
  // to the Node orchestrator below. Its tests, and those of every module,
  // match the _test.py suffix.
  'scripts/release_orchestrator_asset_releases.py',
  'scripts/release_orchestrator_driver.py',
  'scripts/release_orchestrator_model.py',
  'scripts/release_orchestrator_native.py',
  'scripts/release_orchestrator_planner.py',
  'scripts/release_orchestrator_release_tags.py',
  'scripts/release_orchestrator_run_names.py',
  'scripts/release_orchestrator_stage_proofs.py',
  'scripts/release_orchestrator_transport.py',
  'scripts/release_orchestrator_workflow_runs.py',
  'scripts/release_publication_state.py',
  'scripts/release_qualification.py',
  // Node ports of the two above. The shared json/cli/errors/contract/manifest
  // modules stay governed, like release_contract.py and
  // generate_release_manifest.py.
  'scripts/release/archive.mjs',
  'scripts/release/publication_state.mjs',
  'scripts/release/qualification.mjs',
  'scripts/release/qualify.mjs',
  'scripts/release/wav.mjs',
  // The Node release orchestrator. Each module is listed explicitly, never by
  // prefix.
  'scripts/release/orchestrator/asset_releases.mjs',
  'scripts/release/orchestrator/cli.mjs',
  'scripts/release/orchestrator/driver.mjs',
  'scripts/release/orchestrator/model.mjs',
  'scripts/release/orchestrator/native.mjs',
  'scripts/release/orchestrator/planner.mjs',
  'scripts/release/orchestrator/release_tags.mjs',
  'scripts/release/orchestrator/run_names.mjs',
  'scripts/release/orchestrator/stage_proofs.mjs',
  'scripts/release/orchestrator/transport.mjs',
  'scripts/release/orchestrator/workflow_runs.mjs',
  // The speech gate's audio pin and transcript, read by
  // release_qualification.py and the speech smoke. Like both of them it
  // decides how a candidate is qualified and is never in the artifact.
  'scripts/speech_to_text_fixture.json',
  // The Python orchestrator's CLI entry; see the modules above.
  'scripts/stable_release_orchestrator.py',
  // Ported to scripts/verify_ci_reliability.mjs; the .py path was deleted.
  'scripts/verify_ci_reliability.mjs',
  'scripts/verify_ci_reliability.py',
  // Ported to tests/js/*_api_contract_test.mjs, which the _test.mjs suffix
  // covers; the .py paths were deleted.
  'scripts/verify_decision_api.py',
  'scripts/verify_state_persistence_api.py',
  'scripts/verify_text_to_speech_api.py',
  // Current home of the scripts/bridge_operation_queue_* fixtures above,
  // whose scripts/ paths were deleted.
  'tests/js/bridge_operation_queue_direct_cases.mjs',
  'tests/js/bridge_operation_queue_fixtures.mjs',
  'tests/js/bridge_operation_queue_lifecycle_contract_cases.mjs',
  'tests/js/bridge_operation_queue_worker_proxy_cases.mjs',
  // Source readers the JS contract tests import.
  'tests/js/bridge_js_source.mjs',
  'tests/js/native_core_source.mjs',
]));
// The release tooling's Node tests and their fixtures never reach a build.
export const ORCHESTRATION_ONLY_PREFIXES = Object.freeze(['docs/', 'tests/release/']);
export const ORCHESTRATION_ONLY_SCRIPT_SUFFIXES = Object.freeze([
  // Node ports of the Python browser smokes. Keep the .py suffix: the
  // commits that port a smoke delete its .py path, and history must not
  // classify those commits as build inputs.
  '_browser_smoke.mjs',
  '_browser_smoke.py',
  '_test.mjs',
  '_test.py',
]);
export const ORCHESTRATION_ONLY_JS_TEST_SUFFIX = '_test.mjs';

// Return whether a repository path can change published bridge artifacts.
//
// The default is deliberately governed. Only known workflow, validation,
// test, and documentation surfaces are exempt, so deleting this classifier or
// adding a new build input makes the identity advance rather than silently
// reusing an older immutable release.
export function isGovernedBridgePath(path) {
  if (
    typeof path !== 'string'
    || !path
    || path.startsWith('/')
    || path.includes('\\')
    || path.split('/').some((part) => part === '' || part === '.' || part === '..')
  ) {
    throw new ContractError(`git reported an invalid repository path: ${pyRepr(path)}`);
  }
  if (ORCHESTRATION_ONLY_PATHS.has(path)) return false;
  if (ORCHESTRATION_ONLY_PREFIXES.some((prefix) => path.startsWith(prefix))) return false;
  if (path.startsWith('scripts/') && ORCHESTRATION_ONLY_SCRIPT_SUFFIXES.some((suffix) => path.endsWith(suffix))) return false;
  if (path.startsWith('tests/') && path.endsWith(ORCHESTRATION_ONLY_JS_TEST_SUFFIX)) return false;
  return true;
}

// Current executable source plus its governed runtime/build identity.
export class BridgeSourceIdentity {
  constructor({ bridgeSourceSha, bridgeBuildSha }) {
    this.bridgeSourceSha = bridgeSourceSha;
    this.bridgeBuildSha = bridgeBuildSha;
    Object.freeze(this);
  }
}

// str(subprocess.CalledProcessError): "Command '(...)' returned non-zero exit
// status N." or "... died with <Signals.SIGX: n>.".
function calledProcessErrorText(argv, completed) {
  const command = `(${argv.map((arg) => pyRepr(arg)).join(', ')}${argv.length === 1 ? ',' : ''})`;
  if (completed.signal) {
    const number = os.constants.signals[completed.signal];
    return `Command '${command}' died with <Signals.${completed.signal}: ${number}>.`;
  }
  return `Command '${command}' returned non-zero exit status ${completed.status}.`;
}

// _git_output: subprocess.run(("git", "-C", repository, *args), check=True,
// text=True).stdout, with a failure as ContractError.
export function gitOutput(repository, ...args) {
  const argv = ['git', '-C', String(repository), ...args];
  const completed = spawnSync(argv[0], argv.slice(1), {
    stdio: ['inherit', 'pipe', 'pipe'],
    maxBuffer: 1024 * 1024 * 1024,
  });
  let detail = '';
  let fallback = '';
  if (completed.error) {
    if (!isOSError(completed.error)) throw completed.error;
    fallback = osErrorString(completed.error, 'git');
  } else if (completed.status !== 0) {
    detail = pyStrip(pyUniversalNewlines(pyDecodeUtf8(completed.stderr)));
    fallback = calledProcessErrorText(argv, completed);
  } else {
    return pyUniversalNewlines(pyDecodeUtf8(completed.stdout));
  }
  throw new ContractError(`could not resolve governed bridge source with git: ${detail || fallback}`);
}

// Resolve HEAD and the newest first-parent governed build change.
//
// The scheduled checkout must have complete history. Walking first-parent
// makes the selected build identity a commit on the exact default-branch
// line, including merge commits that introduce governed files. `repository`
// is a path string.
export function resolveBridgeSourceIdentity(repository, head = 'HEAD') {
  const root = pyPath(String(repository));
  if (!pyIsDir(root) || pyIsSymlink(root)) throw new ContractError(`bridge repository is not a directory: ${root}`);
  const sourceSha = pyStrip(gitOutput(root, 'rev-parse', '--verify', `${head}^{commit}`));
  if (pyFullmatch(COMMIT_RE, sourceSha) === null) {
    throw new ContractError('resolved bridge source is not a full lowercase commit SHA');
  }
  const history = gitOutput(
    root,
    'log',
    '--first-parent',
    '--format=%x00%H%x00',
    '--name-only',
    '--no-renames',
    sourceSha,
  );
  const fields = history.split('\0');
  if (fields.length === 0 || fields[0] !== '' || fields.length < 3 || fields.length % 2 === 0) {
    throw new ContractError('git returned malformed bridge first-parent history');
  }
  for (let index = 1; index < fields.length; index += 2) {
    const commit = fields[index];
    if (pyFullmatch(COMMIT_RE, commit) === null) throw new ContractError('git returned a malformed first-parent commit');
    const changedPaths = pySplitlines(fields[index + 1]).filter((path) => path);
    if (changedPaths.some((path) => isGovernedBridgePath(path))) {
      return new BridgeSourceIdentity({ bridgeSourceSha: sourceSha, bridgeBuildSha: commit });
    }
  }
  throw new ContractError('bridge history contains no governed runtime/build source');
}

// Keep untrusted callers from turning the environment PAT into a deputy.
//
// Scheduled executions are authorized by the trusted default-branch workflow.
// A workflow_run continuation and a manual dispatch additionally require both
// GitHub actor identities to be the repository owner before any
// environment-scoped credential is used. A workflow_run event always executes
// the default-branch workflow definition, so the continuation cannot be
// redefined from a pull request or a fork.
export function requireOrchestrationCaller(eventName, actor, triggeringActor) {
  if (eventName === 'schedule') return;
  if (
    (eventName === 'workflow_dispatch' || eventName === 'workflow_run')
    && actor === REPOSITORY_OWNER
    && triggeringActor === REPOSITORY_OWNER
  ) {
    return;
  }
  throw new ContractError(
    'stable orchestration requires a schedule event, or an owner-initiated '
    + 'workflow_dispatch or workflow_run continuation with owner actor and '
    + 'triggering_actor',
  );
}

export function renderStepSummary(plan) {
  const target = plan.releaseTarget;
  const provenance = plan.provenance;
  const lines = [
    '### Stable Web bridge release orchestration',
    '',
    `- Action: \`${plan.action}\``,
    `- Reason: ${plan.reason}`,
    `- Correlation: \`${plan.correlationId}\``,
    `- Bridge source: \`${provenance.bridgeSourceSha}\``,
    `- Governed bridge build: \`${provenance.bridgeBuildSha}\``,
    `- llama.cpp: \`${provenance.upstreamTag}@${provenance.upstreamCommit}\``,
    `- Native release: \`${provenance.nativeRepo}@${provenance.nativeReleaseTag}\``,
    `- Native manifest SHA-256: \`${provenance.nativeManifestSha256}\``,
  ];
  if (target !== null) {
    lines.push(`- Output release: \`${pyStr(target.releaseTag)}\` (rebuild \`${pyStr(target.releaseRebuild)}\`)`);
  }
  if (plan.candidateRunId) lines.push(`- Candidate run: \`${plan.candidateRunId}\``);
  if (plan.qualificationRunId) lines.push(`- Qualification run: \`${plan.qualificationRunId}\``);
  if (plan.inFlightWorkflow) lines.push(`- In flight: \`${plan.inFlightWorkflow}\` run \`${pyStr(plan.inFlightRunId)}\``);
  if (plan.dispatchWorkflow) {
    lines.push(
      `- Dispatched: \`${plan.dispatchWorkflow}\` at \`${pyStr(plan.dispatchRef)}\` `
      + `as \`${pyStr(plan.dispatchRunName)}\` (run \`${pyStr(plan.dispatchedRunId)}\`)`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

const FAILING_ACTIONS = Object.freeze(new Set([OrchestrationAction.BLOCKED]));

const required = (flag, type = 'str') => ({ flag, required: true, type });
const optional = (flag, type = 'str', fallback = null) => ({
  flag, required: false, type, default: fallback,
});

// Each subcommand's options, as parseArguments takes them, and each option's
// argparse `choices`.
export const COMMANDS = Object.freeze({
  'scan-native': {
    options: [
      required('--manifest', 'path'),
      required('--native-release-tag'),
      required('--bridge-source-sha'),
      required('--bridge-build-sha'),
      required('--native-release-published-at'),
      required('--channel'),
      optional('--output-json', 'path'),
    ],
    choices: { channel: ['--channel', [...CHANNELS.keys()].sort(compareCodePoints)] },
  },
  'resolve-bridge-source': {
    options: [required('--repository', 'path'), optional('--head', 'str', 'HEAD'), optional('--output-json', 'path')],
  },
  'select-stable-native-backlog': {
    options: [required('--releases-json', 'path'), optional('--output-json', 'path')],
  },
  orchestrate: {
    options: [
      required('--provenance-json', 'path'),
      required('--workspace', 'path'),
      optional('--output-plan-json', 'path'),
      optional('--step-summary-file', 'path'),
      optional('--dry-run', 'flag'),
    ],
  },
  'orchestrate-backlog': {
    options: [
      required('--provenance-list-json', 'path'),
      required('--workspace', 'path'),
      optional('--output-plan-json', 'path'),
      optional('--step-summary-file', 'path'),
      optional('--dry-run', 'flag'),
    ],
  },
});

const PROG = progName(import.meta.url);

function usageError(prog, message) {
  throw new ArgparseExit(2, { stderr: `usage: ${prog} [-h] {${Object.keys(COMMANDS).join(',')}} ...\n${prog}: error: ${message}\n` });
}

// argparse with add_subparsers(dest="subcommand", required=True), under
// cli.mjs's policy (stricter, never looser): the subcommand must be the first
// argument, and the rest is parsed by parseArguments. Returns
// { subcommand, args } with camelCase keys.
export function parseCommand(argv) {
  if (argv.length === 0) usageError(PROG, 'the following arguments are required: subcommand');
  const [subcommand, ...rest] = argv;
  if (subcommand === '-h' || subcommand === '--help') {
    throw new ArgparseExit(0, { stdout: `usage: ${PROG} [-h] {${Object.keys(COMMANDS).join(',')}} ...\n` });
  }
  if (!Object.hasOwn(COMMANDS, subcommand)) {
    usageError(PROG, `argument subcommand: invalid choice: ${pyRepr(subcommand)} (choose from ${Object.keys(COMMANDS).map(pyRepr).join(', ')})`);
  }
  const spec = COMMANDS[subcommand];
  const prog = `${PROG} ${subcommand}`;
  const args = parseArguments(rest, { prog, options: spec.options });
  for (const [key, [flag, choices]] of Object.entries(spec.choices ?? {})) {
    if (args[key] !== null && !choices.includes(args[key])) {
      throw new ArgparseExit(2, {
        stderr: `usage: ${prog} ...\n${prog}: error: argument ${flag}: invalid choice: ${pyRepr(args[key])} (choose from ${choices.map(pyRepr).join(', ')})\n`,
      });
    }
  }
  return { subcommand, args };
}

// _provenance_to_dict.
export function provenanceToDict(provenance) {
  return {
    bridge_source_sha: provenance.bridgeSourceSha,
    bridge_build_sha: provenance.bridgeBuildSha,
    upstream_tag: provenance.upstreamTag,
    upstream_commit: provenance.upstreamCommit,
    native_repo: provenance.nativeRepo,
    native_release_tag: provenance.nativeReleaseTag,
    native_commit: provenance.nativeCommit,
    native_manifest_sha256: provenance.nativeManifestSha256,
    native_release_published_at: provenance.nativeReleasePublishedAt,
  };
}

const PROVENANCE_FIELDS = Object.freeze([
  ['bridge_source_sha', 'bridgeSourceSha'],
  ['bridge_build_sha', 'bridgeBuildSha'],
  ['upstream_tag', 'upstreamTag'],
  ['upstream_commit', 'upstreamCommit'],
  ['native_repo', 'nativeRepo'],
  ['native_release_tag', 'nativeReleaseTag'],
  ['native_commit', 'nativeCommit'],
  ['native_manifest_sha256', 'nativeManifestSha256'],
  ['native_release_published_at', 'nativeReleasePublishedAt'],
]);

// _load_provenance_payload.
export function loadProvenancePayload(payload, label) {
  if (!isDict(payload) && !(payload instanceof Map)) throw new ContractError(`${label} must be a JSON object`);
  const keys = pyKeys(payload);
  if (keys.length !== PROVENANCE_FIELDS.length || !PROVENANCE_FIELDS.every(([snake]) => keys.includes(snake))) {
    throw new ContractError(`${label} has missing or unexpected fields`);
  }
  const options = Object.fromEntries(PROVENANCE_FIELDS.map(([snake, camel]) => [camel, pyGet(payload, snake)]));
  return requireStableProvenance(new NativeProvenance(options));
}

// _load_provenance: `path` is a path string.
export function loadProvenance(path) {
  const payload = strictJsonLoads(pyReadText(String(path)), 'provenance');
  return loadProvenancePayload(payload, 'provenance');
}

// _load_provenance_backlog: the provenances in (published_at, native order).
export function loadProvenanceBacklog(path) {
  const payload = strictJsonLoads(pyReadText(String(path)), 'provenance backlog');
  if (!Array.isArray(payload)) throw new ContractError('provenance backlog root must be a JSON array');
  const provenances = payload.map((item, index) => loadProvenancePayload(item, `provenance backlog entry ${index}`));
  const correlations = new Set();
  const nativeTags = new Set();
  for (const provenance of provenances) {
    const correlation = computeCorrelationId(provenance);
    if (correlations.has(correlation) || nativeTags.has(provenance.nativeReleaseTag)) {
      throw new ContractError('provenance backlog contains a duplicate pipeline');
    }
    correlations.add(correlation);
    nativeTags.add(provenance.nativeReleaseTag);
  }
  const keyed = provenances.map((value) => [value.nativeReleasePublishedAt, nativeReleaseOrder(value.nativeReleaseTag), value]);
  keyed.sort((left, right) => compareCodePoints(left[0], right[0]) || pyCompareIntTuples(left[1], right[1]));
  return keyed.map(([, , value]) => value);
}

// Path.write_text(text, encoding="utf-8").
function writeText(path, text) {
  try {
    fs.writeFileSync(path, text, 'utf8');
  } catch (error) {
    throw pyOSError(error, path);
  }
}

// open(path, "a", encoding="utf-8").write(text).
function appendText(path, text) {
  try {
    fs.appendFileSync(path, text, 'utf8');
  } catch (error) {
    throw pyOSError(error, path);
  }
}

function defaultCreateGateway({ readToken, dispatchToken, env }) {
  return new GhGateway({ readToken, dispatchToken, env });
}

export function main(argv = process.argv.slice(2), {
  env = process.env,
  createGateway = defaultCreateGateway,
  stdout = (text) => process.stdout.write(text),
  stderr = (text) => process.stderr.write(text),
} = {}) {
  const { subcommand, args } = parseCommand(argv);
  const print = (text) => stdout(`${text}\n`);

  if (subcommand === 'scan-native') {
    const provenance = scanNativeProvenance({
      manifestPath: args.manifest,
      nativeReleaseTag: args.nativeReleaseTag,
      bridgeSourceSha: args.bridgeSourceSha,
      bridgeBuildSha: args.bridgeBuildSha,
      channel: args.channel,
      nativeReleasePublishedAt: args.nativeReleasePublishedAt,
    });
    const payload = pyJsonDumps(provenanceToDict(provenance), { indent: 2, sortKeys: true });
    if (args.outputJson) writeText(args.outputJson, `${payload}\n`);
    print(payload);
    return 0;
  }

  if (subcommand === 'resolve-bridge-source') {
    const identity = resolveBridgeSourceIdentity(args.repository, args.head);
    const payload = pyJsonDumps({
      bridge_source_sha: identity.bridgeSourceSha,
      bridge_build_sha: identity.bridgeBuildSha,
    }, { indent: 2, sortKeys: true });
    if (args.outputJson) writeText(args.outputJson, `${payload}\n`);
    print(payload);
    return 0;
  }

  if (subcommand === 'select-stable-native-backlog') {
    const releases = strictJsonLoads(pyReadText(String(args.releasesJson)), 'native release listing');
    if (!Array.isArray(releases)) throw new ContractError('native release listing must be a JSON array');
    const selected = selectStableNativeBacklog(releases);
    const payload = pyJsonDumps(selected, { indent: 2 });
    if (args.outputJson) writeText(args.outputJson, `${payload}\n`);
    print(payload);
    return 0;
  }

  requireOrchestrationCaller(env.GITHUB_EVENT_NAME ?? '', env.GITHUB_ACTOR ?? '', env.GITHUB_TRIGGERING_ACTOR ?? '');
  const gateway = createGateway({
    readToken: env.GH_TOKEN ?? '',
    dispatchToken: env.WEBGPU_BRIDGE_ASSETS_PAT ?? null,
    env,
  });
  makeDirectories(args.workspace);

  if (subcommand === 'orchestrate-backlog') {
    const provenances = loadProvenanceBacklog(args.provenanceListJson);
    const plans = [];
    const errors = [];
    const reservedReleaseTags = new Set();
    const satisfiedCorrelationIds = new Set();
    let publicationBarrierNativeTag = null;
    let newestNativeOrder = [];
    for (const provenance of provenances) {
      const order = nativeReleaseOrder(provenance.nativeReleaseTag);
      if (pyCompareIntTuples(order, newestNativeOrder) > 0) newestNativeOrder = order;
    }
    for (const provenance of provenances) {
      const correlationId = computeCorrelationId(provenance);
      const pipelineWorkspace = pyJoinPath(args.workspace, correlationId);
      makeDirectories(pipelineWorkspace);
      try {
        const plan = advancePipeline(gateway, {
          provenance,
          workspace: pipelineWorkspace,
          dryRun: args.dryRun,
          reservedReleaseTags,
          satisfiedCorrelationIds,
          publicationAllowed: publicationBarrierNativeTag === null,
          publicationBarrierNativeTag,
          newerNativeScanned: pyCompareIntTuples(nativeReleaseOrder(provenance.nativeReleaseTag), newestNativeOrder) < 0,
        });
        plans.push(plan.toDict());
        if (plan.action === OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE) {
          satisfiedCorrelationIds.add(correlationId);
        } else if (plan.releaseTarget !== null) {
          reservedReleaseTags.add(plan.releaseTarget.releaseTag);
        }
        if (publicationBarrierNativeTag === null && ![
          OrchestrationAction.NOOP,
          OrchestrationAction.SUPERSEDED,
          OrchestrationAction.SATISFIED_BY_IDENTICAL_RELEASE,
        ].includes(plan.action)) {
          publicationBarrierNativeTag = provenance.nativeReleaseTag;
        }
        if (args.stepSummaryFile) appendText(args.stepSummaryFile, `${renderStepSummary(plan)}\n`);
      } catch (error) {
        if (!(error instanceof ContractError)) throw error;
        errors.push({
          correlation_id: correlationId,
          native_release_tag: provenance.nativeReleaseTag,
          error: error.message,
        });
        if (args.stepSummaryFile) {
          appendText(
            args.stepSummaryFile,
            '### Blocked stable Web bridge release orchestration\n\n'
            + `- Correlation: \`${correlationId}\`\n`
            + `- Native release: \`${provenance.nativeReleaseTag}\`\n`
            + `- Error: ${error.message}\n\n`,
          );
        }
        // A transport/readback error can mean a dispatch occurred but its
        // state is not yet observable. Stop before another backlog entry can
        // claim a colliding output identity under uncertainty.
        break;
      }
    }
    const result = { schema_version: 1, plans, errors };
    const payload = pyJsonDumps(result, { indent: 2, sortKeys: true });
    if (args.outputPlanJson) writeText(args.outputPlanJson, `${payload}\n`);
    print(payload);
    const blocked = plans.filter((plan) => FAILING_ACTIONS.has(plan.action));
    if (errors.length > 0 || blocked.length > 0) {
      stderr('error: one or more stable pipelines are blocked\n');
      return 1;
    }
    return 0;
  }

  const provenance = loadProvenance(args.provenanceJson);
  const plan = advancePipeline(gateway, {
    provenance,
    workspace: args.workspace,
    dryRun: args.dryRun,
  });
  const payload = pyJsonDumps(plan.toDict(), { indent: 2, sortKeys: true });
  if (args.outputPlanJson) writeText(args.outputPlanJson, `${payload}\n`);
  if (args.stepSummaryFile) appendText(args.stepSummaryFile, `${renderStepSummary(plan)}\n`);
  print(payload);
  if (FAILING_ACTIONS.has(plan.action)) {
    stderr(`error: ${plan.reason}\n`);
    return 1;
  }
  return 0;
}

// Run main() as `python3 scripts/stable_release_orchestrator.py` runs: a
// ContractError prints "error: <message>" and exits 1 (the __main__ block's
// SystemExit), an argparse exit prints its text with its status, and any
// other Python exception prints its traceback's last line and exits 1.
// Returns the exit status.
export function runMain(argv = process.argv.slice(2), options = {}) {
  const stdout = options.stdout ?? ((text) => process.stdout.write(text));
  const stderr = options.stderr ?? ((text) => process.stderr.write(text));
  try {
    const status = main(argv, { ...options, stdout, stderr });
    return Number.isInteger(status) ? status : 0;
  } catch (error) {
    if (error instanceof ArgparseExit) {
      if (error.stdout) stdout(error.stdout);
      if (error.stderr) stderr(error.stderr);
      return error.status;
    }
    if (error instanceof ContractError) {
      stderr(`error: ${error.message}\n`);
      return 1;
    }
    if (error instanceof PyException) {
      stderr(`Traceback (most recent call last):\n${error.tracebackLine}\n`);
      return 1;
    }
    throw error;
  }
}

if (import.meta.main) process.exitCode = runMain();
