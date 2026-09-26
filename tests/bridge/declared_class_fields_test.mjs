import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import ts from 'typescript';

// A class field without an initializer must be `declare`d. Under ES2022 class
// fields a plain `x: T;` defines the property as undefined before the
// constructor runs, which changes behaviour rather than only adding a type.
// Walk the syntax tree so modifiers, multi-line and function types, and
// comments cannot hide a field from the check.

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const sourceDir = path.join(rootDir, 'js/src');

function typeScriptSources(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      return typeScriptSources(entryPath);
    }
    return /\.[cm]?ts$/.test(entry.name) && !entry.name.endsWith('.d.ts') ? [entryPath] : [];
  });
}

function undeclaredFields(fileName, text) {
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const found = [];
  const visit = (node) => {
    if (ts.isPropertyDeclaration(node) && node.initializer === undefined) {
      const hasModifier = (target, kind) => (ts.getModifiers(target) || [])
        .some((modifier) => modifier.kind === kind);
      // Abstract fields and members of an ambient `declare class` emit nothing.
      const emitsNothing = hasModifier(node, ts.SyntaxKind.DeclareKeyword)
        || hasModifier(node, ts.SyntaxKind.AbstractKeyword)
        || hasModifier(node.parent, ts.SyntaxKind.DeclareKeyword);
      if (!emitsNothing) {
        const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
        found.push(`${path.relative(rootDir, fileName)}:${line + 1} ${node.name.getText(source)}`);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

const sources = typeScriptSources(sourceDir);
assert.ok(sources.length > 0, 'expected TypeScript sources under js/src');
const offenders = sources.flatMap((fileName) => undeclaredFields(fileName, readFileSync(fileName, 'utf8')));
assert.deepEqual(offenders, [], 'class fields without an initializer must use `declare`');

// The walker itself: every form a line-based check misses must be caught.
const probe = undeclaredFields(path.join(rootDir, 'probe.ts'), [
  'class A {',
  '  declare ok: number;',
  '  withInit = 1;',
  '  static withStaticInit = 2;',
  '  plain: number;',
  '  private priv: string;',
  '  protected override prot?: string;',
  '  accessor acc: number;',
  '  commented: number; // note',
  '  fn: (value: number) => void',
  '  multi:',
  '    | string',
  '    | null;',
  '}',
  'export default class { anon: boolean; }',
  'abstract class B { abs: number; abstract skipped: number; }',
  'declare class Ambient { skipped: number; }',
  'const C = class { expr: number; };',
  'interface NotAClass { field: number; }',
  'const literal = { key: 1 };',
].join('\n'));
assert.deepEqual(
  probe.map((entry) => entry.split(' ')[1]),
  ['plain', 'priv', 'prot', 'acc', 'commented', 'fn', 'multi', 'anon', 'abs', 'expr'],
);

console.log(`Declared class field checks passed (${sources.length} TypeScript sources)`);
