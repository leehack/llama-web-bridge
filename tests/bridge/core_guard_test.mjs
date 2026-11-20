import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

import { guardCore, isCoreRetired, retireCore } from '../../js/src/internal/core_guard.ts';
import { EmbeddingError, ModelStagingError } from '../../js/src/llama_webgpu_bridge.js';

function check(label, condition, detail = '') {
  assert.ok(condition, detail ? `${label}: ${detail}` : label);
}

function rawCore() {
  const counts = { total: 0, retireHook: 0 };
  let trap = null;
  const call = (name) => {
    counts.total += 1;
    counts[name] = (counts[name] || 0) + 1;
    if (trap && trap.name === name) {
      const error = trap.make();
      if (trap.async) {
        return Promise.reject(error);
      }
      throw error;
    }
    return 0;
  };
  const core = {
    HEAPU8: new Uint8Array(4),
    ccall: (name) => call(name),
    cwrap: (name) => () => call(name),
    stackSave: () => call('stackSave'),
    stackAlloc: () => call('stackAlloc'),
    stackRestore: () => call('stackRestore'),
    _malloc: () => call('_malloc'),
    _free: () => call('_free'),
    _llamadart_x: () => call('_llamadart_x'),
    llamadartJsModelFsWriteDirect: () => call('writeDirect'),
    llamadartRetire: () => {
      counts.retireHook += 1;
    },
    FS: {
      writeFile() {
        return call('FS.writeFile');
      },
      unlink() {
        return call('FS.unlink');
      },
    },
  };
  return { core, counts, setTrap: (value) => { trap = value; } };
}

const wasmTrap = () => new WebAssembly.RuntimeError('unreachable');

test('guard: access styles', async () => {
  for (const [label, trigger, name] of [
    ['direct ccall', (g) => g.ccall('a'), 'a'],
    ['computed access', (g) => g['cc' + 'all']('a'), 'a'],
    ['destructured', (g) => { const { ccall } = g; return ccall('a'); }, 'a'],
    ['raw export', (g) => g._llamadart_x(), '_llamadart_x'],
    ['stack helper', (g) => g.stackAlloc(8), 'stackAlloc'],
    ['FS method', (g) => g.FS.writeFile('/x', new Uint8Array(1)), 'FS.writeFile'],
    ['FS passed to a helper', (g) => ((fs) => fs.writeFile('/x'))(g.FS), 'FS.writeFile'],
    ['write-direct alias', (g) => g.llamadartJsModelFsWriteDirect(1), 'writeDirect'],
  ]) {
    const { core, counts, setTrap } = rawCore();
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const alias = guarded.ccall;
    const stored = guarded.cwrap('b');
    const fsAlias = guarded.FS.unlink;
    const trapError = wasmTrap();
    setTrap({ name, make: () => trapError });
    let thrown = null;
    try {
      trigger(guarded);
    } catch (error) {
      thrown = error;
    }
    const callsAtRetire = counts.total;
    const later = [];
    for (const use of [() => alias('z'), () => stored(), () => fsAlias('/x'), () => guarded.ccall('z'),
      () => guarded._malloc(1), () => guarded.FS.writeFile('/y')]) {
      try {
        use();
        later.push('ran');
      } catch (error) {
        later.push(error === trapError ? 'cause' : 'other');
      }
    }
    check(`${label}: same error rethrown, retired once, hook once, no raw call afterwards`,
      thrown === trapError && retired === 1 && counts.retireHook === 1 && counts.total === callsAtRetire
      && later.every((entry) => entry === 'cause') && isCoreRetired(guarded),
      JSON.stringify({ retired, hook: counts.retireHook, later, extra: counts.total - callsAtRetire }));
  }
});

test('guard: signals', async () => {
  {
    const { core, setTrap } = rawCore();
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const trapError = wasmTrap();
    setTrap({ name: 'a', async: true, make: () => trapError });
    const rejection = await guarded.ccall('a').catch((error) => error);
    check('async rejection retires and keeps the error', rejection === trapError && retired === 1);
  }
  // A core that runs in another realm (a same-origin iframe). Its traps carry
  // that realm's RuntimeError, as probe_trap_realm.mjs shows for a real instance.
  const TRAP_WASM = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0,
    7, 8, 1, 4, 116, 114, 97, 112, 0, 0, 10, 5, 1, 3, 0, 0, 11]);
  function foreignRealm() {
    return vm.runInContext(`({
      RuntimeError: WebAssembly.RuntimeError,
      trap: new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports.trap,
      abortError: () => new WebAssembly.RuntimeError('Aborted()'),
      // What ordinary code in that realm can throw without being a trap.
      lookalike: () => new (class RuntimeError extends Error {})('ordinary error'),
    })`, vm.createContext({ bytes: TRAP_WASM }));
  }
  {
    const realm = foreignRealm();
    const { core, counts } = rawCore();
    core.llamadartWasmRuntimeError = realm.RuntimeError;
    core.ccall = (name) => {
      counts.total += 1;
      return name === 'trap' ? realm.trap() : 0;
    };
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    let thrown;
    try { guarded.ccall('trap'); } catch (error) { thrown = error; }
    check('a real trap in a foreign-realm core retires through the class the core exports',
      retired === 1 && !(thrown instanceof WebAssembly.RuntimeError) && !(thrown instanceof Error)
      && thrown instanceof realm.RuntimeError && isCoreRetired(guarded));
  }
  {
    const realm = foreignRealm();
    const { core, setTrap } = rawCore();
    core.llamadartWasmRuntimeError = realm.RuntimeError;
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const abortError = realm.abortError();
    setTrap({ name: 'a', make: () => abortError });
    let thrown;
    try { guarded.ccall('a'); } catch (error) { thrown = error; }
    check('a foreign-realm abort() error retires', thrown === abortError && retired === 1);
  }
  {
    // An older core exports no class. Only onAbort can retire it across realms.
    const realm = foreignRealm();
    const { core, counts } = rawCore();
    core.ccall = (name) => {
      counts.total += 1;
      return name === 'trap' ? realm.trap() : 0;
    };
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    let thrown;
    try { guarded.ccall('trap'); } catch (error) { thrown = error; }
    check('without the exported class a foreign-realm trap does not retire (known limit)',
      retired === 0 && thrown?.name === 'RuntimeError' && !isCoreRetired(guarded));
    retireCore(guarded, 'native code called abort()');
    check('onAbort still retires that core', retired === 1);
  }
  for (const [label, make, exported] of [
    ['a page-realm class named RuntimeError', () => new (class RuntimeError extends Error {})('ordinary error'), false],
    ['a foreign-realm class named RuntimeError, core exports its real class',
      (realm) => realm.lookalike(), true],
    ['a foreign-realm class named RuntimeError, core exports nothing', (realm) => realm.lookalike(), false],
  ]) {
    const realm = foreignRealm();
    const { core, counts, setTrap } = rawCore();
    if (exported) {
      core.llamadartWasmRuntimeError = realm.RuntimeError;
    }
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const value = make(realm);
    setTrap({ name: 'a', make: () => value });
    let thrown;
    try { guarded.ccall('a'); } catch (error) { thrown = error; }
    setTrap(null);
    try { guarded.ccall('b'); } catch (_) { /* a retired guard rethrows; the check below reports it */ }
    check(`${label} does not retire`, thrown === value && value.constructor.name === 'RuntimeError'
      && retired === 0 && counts.b === 1 && !isCoreRetired(guarded));
  }
  for (const [label, exported] of [
    ['an arrow function', () => {}],
    ['a number', 7],
    ['an object', { prototype: {} }],
  ]) {
    const { core, counts, setTrap } = rawCore();
    core.llamadartWasmRuntimeError = exported;
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const plain = new Error('plain');
    setTrap({ name: 'a', make: () => plain });
    let thrown;
    try { guarded.ccall('a'); } catch (error) { thrown = error; }
    const trapError = wasmTrap();
    setTrap({ name: 'a', make: () => trapError });
    let second;
    try { guarded.ccall('a'); } catch (error) { second = error; }
    check(`an exported class that is ${label} is ignored; page-realm traps still retire`,
      thrown === plain && second === trapError && retired === 1 && counts.a === 2);
  }
  {
    const { core, counts } = rawCore();
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    retireCore(guarded, 'native code called abort()');
    retireCore(guarded, 'again');
    let message = '';
    try { guarded.ccall('a'); } catch (error) { message = error.message; }
    check('onAbort alone retires once and later calls never reach the core',
      retired === 1 && counts.total === 0 && counts.retireHook === 1 && /^Aborted\(native code called abort\(\)\)/.test(message),
      message);
  }
  {
    const { core, setTrap } = rawCore();
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    // abort(): onAbort first, then the RuntimeError the caller sees.
    const trapError = wasmTrap();
    setTrap({ name: 'a', make: () => { retireCore(guarded, 'x'); return trapError; } });
    let thrown;
    try { guarded.ccall('a'); } catch (error) { thrown = error; }
    let later;
    try { guarded.ccall('b'); } catch (error) { later = error; }
    check('both signals: one retirement, later calls throw the RuntimeError', retired === 1
      && thrown === trapError && later === trapError);
  }
  for (const [label, make] of [
    ["Error('operation aborted')", () => new Error('operation aborted')],
    ["Error('Aborted()')", () => new Error('Aborted()')],
    ['thrown string', () => 'aborted'],
    ['AbortError', () => new DOMException('aborted', 'AbortError')],
    ['ModelStagingError', () => new ModelStagingError('MODEL_STAGING_MOUNT_FAILED', 'aborted(')],
    ['EmbeddingError', () => new EmbeddingError('EMBEDDING_FAILED', 'aborted')],
    ['ErrnoError-like', () => Object.assign(Object.create({ constructor: { name: 'ErrnoError' } }), { errno: 44 })],
    ['RangeError', () => new RangeError('Maximum call stack size exceeded')],
  ]) {
    const { core, counts, setTrap } = rawCore();
    let retired = 0;
    const guarded = guardCore(core, () => { retired += 1; });
    const value = make();
    setTrap({ name: 'a', make: () => value });
    let thrown;
    try { guarded.ccall('a'); } catch (error) { thrown = error; }
    setTrap(null);
    guarded.ccall('b');
    check(`${label} does not retire`, thrown === value && retired === 0 && counts.b === 1 && !isCoreRetired(guarded));
  }
  {
    const { core } = rawCore();
    const guarded = guardCore(core, () => {});
    check('values pass through, wrappers are stable', guarded.HEAPU8 === core.HEAPU8
      && guarded.ccall === guarded.ccall && guarded.FS === guarded.FS && guarded.FS.unlink === guarded.FS.unlink);
  }
});

test('a custom core with a locked ccall property is outside the factory contract', () => {
  const core = Object.freeze({ ccall: () => 7 });
  const guarded = guardCore(core, () => assert.fail('a proxy invariant is not a native abort'));
  assert.throws(() => guarded.ccall('test'), {
    name: 'TypeError',
    message: /ccall.*read-only.*non-configurable/,
  });
  assert.equal(isCoreRetired(guarded), false);
});

test('the native core exports its own realm RuntimeError', () => {
  const source = readFileSync(new URL('../../src/llama_webgpu_core_post.js', import.meta.url), 'utf8');
  assert.match(source, /Module\['llamadartWasmRuntimeError'\] = WebAssembly\.RuntimeError;/);
});

test('the runtime only adopts guarded factory results', () => {
  const root = new URL('../../js/src/', import.meta.url);
  let factoryCalls = 0;
  let adoptions = 0;
  const isCallTo = (node, name) => ts.isCallExpression(node)
    && ts.isIdentifier(node.expression) && node.expression.text === name;
  for (const file of readdirSync(root, { recursive: true }).filter((name) => /\.[jt]s$/.test(name))) {
    const source = ts.createSourceFile(file, readFileSync(new URL(file, root), 'utf8'),
      ts.ScriptTarget.Latest, true);
    const visit = (node) => {
      if (isCallTo(node, 'moduleFactory')) {
        factoryCalls += 1;
        for (let parent = node.parent; parent; parent = parent.parent) {
          if (isCallTo(parent, 'guardCore')) return;
        }
        assert.fail(`unguarded core factory in ${file}`);
      }
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken
          && ts.isPropertyAccessExpression(node.left) && node.left.expression.kind === ts.SyntaxKind.ThisKeyword
          && node.left.name.text === '_core' && node.right.kind !== ts.SyntaxKind.NullKeyword) {
        adoptions += 1;
        assert.ok(ts.isIdentifier(node.right) && node.right.text === 'candidateCore', file);
        assert.ok(ts.isTryStatement(node.parent.parent.parent), file);
        const bindings = node.parent.parent.statements.filter((statement) => ts.isExpressionStatement(statement)
          && ts.isBinaryExpression(statement.expression)
          && ts.isIdentifier(statement.expression.left) && statement.expression.left.text === 'candidateCore');
        assert.equal(bindings.length, 1, file);
        assert.ok(isCallTo(bindings[0].expression.right, 'guardCore'), file);
        assert.ok(bindings[0].pos < node.pos, file);
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  assert.equal(factoryCalls, 1);
  assert.equal(adoptions, 1);
});
