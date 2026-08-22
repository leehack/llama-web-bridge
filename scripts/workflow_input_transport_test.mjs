#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { parseAllDocuments } from "yaml";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// Fail closed on every input-context spelling, including mixed dotted/bracket
// dereferences such as github.event['inputs'] and github['event']['inputs'].
const INPUT_CONTEXT = /\binputs\b/i;

function resolvedRunScripts(source, sourceName) {
  const scripts = [];
  const documents = parseAllDocuments(source, { merge: true });
  for (const document of documents) {
    if (document.errors.length > 0) {
      throw new Error(`${sourceName}: invalid YAML: ${document.errors[0].message}`);
    }
    const root = document.toJS({ maxAliasCount: 100 });
    const visit = (value) => {
      if (Array.isArray(value)) {
        value.forEach(visit);
        return;
      }
      if (value === null || typeof value !== "object") {
        return;
      }
      for (const [key, child] of Object.entries(value)) {
        if (key === "run") {
          if (typeof child !== "string") {
            throw new Error(`${sourceName}: resolved run value must be a string`);
          }
          scripts.push(child);
        }
        visit(child);
      }
    };
    visit(root);
  }
  return scripts;
}

function unsafeRunScripts(source, sourceName) {
  return resolvedRunScripts(source, sourceName).filter(
    (script) => script.includes("${{") && INPUT_CONTEXT.test(script),
  );
}

const safeFixture = `
env:
  RELEASE_TAG: \${{ inputs.release_tag }}
steps:
  - run: echo "\${RELEASE_TAG}"
`;

const adversarialFixtures = [
  `steps:\n  - run: echo "\${{ inputs.release_tag }}"\n`,
  `steps:\n  - 'run': |\n      echo "\${{ inputs.release_tag }}"\n`,
  `steps:\n  - "run": |2-\n      echo "\${{ inputs['release_tag'] }}"\n`,
  `steps:\n  - ? run\n    : >-\n      echo "\${{ github.event.inputs.release_tag }}"\n`,
  `steps:\n  - { run: 'echo "\${{ inputs.release_tag }}"' }\n`,
  `script: &script |\n  echo "\${{ inputs.release_tag }}"\nsteps:\n  - run: *script\n`,
  `steps:\n  - run: |\n      echo "\${{ format('}}', inputs.release_tag) }}"\n`,
  `steps:\n  - run: echo "\${{ github['event']['inputs']['release_tag'] }}"\n`,
  `steps:\n  - run: echo "\${{ github.event['inputs']['release_tag'] }}"\n`,
];

const errors = [];
if (unsafeRunScripts(safeFixture, "safe fixture").length !== 0) {
  errors.push("env transport fixture must remain allowed");
}
for (const [index, fixture] of adversarialFixtures.entries()) {
  try {
    if (unsafeRunScripts(fixture, `adversarial fixture ${index + 1}`).length !== 1) {
      errors.push(`adversarial fixture ${index + 1} bypassed resolved YAML inspection`);
    }
  } catch (error) {
    errors.push(String(error));
  }
}

const workflowDirectory = path.join(ROOT, ".github", "workflows");
for (const name of fs.readdirSync(workflowDirectory).sort()) {
  if (!name.endsWith(".yml") && !name.endsWith(".yaml")) {
    continue;
  }
  const source = fs.readFileSync(path.join(workflowDirectory, name), "utf8");
  try {
    const unsafe = unsafeRunScripts(source, name);
    if (unsafe.length > 0) {
      errors.push(
        `${name}: ${unsafe.length} run block(s) directly interpolate dispatch inputs; transport through env`,
      );
    }
  } catch (error) {
    errors.push(String(error));
  }
}

if (errors.length > 0) {
  for (const error of errors) {
    console.error(`ERROR: ${error}`);
  }
  process.exit(1);
}

console.log("Workflow input transport contract passed");
