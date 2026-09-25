import { DIRECT_CASES } from './bridge_operation_queue_direct_cases.mjs';
import { WORKER_PROXY_CASES } from './bridge_operation_queue_worker_proxy_cases.mjs';
import { LIFECYCLE_CONTRACT_CASES } from './bridge_operation_queue_lifecycle_contract_cases.mjs';

const CASES = [
  ...DIRECT_CASES,
  ...WORKER_PROXY_CASES,
  ...LIFECYCLE_CONTRACT_CASES,
]
  .sort(([left], [right]) => left - right)
  .map(([, name, run]) => [name, run]);

async function runCaseWithWatchdog(run) {
  let timer = null;
  try {
    return await Promise.race([
      run(),
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error('timed out after 2000ms; the operation queue deadlocked')),
          2000,
        );
      }),
    ]);
  } finally {
    if (timer != null) {
      clearTimeout(timer);
    }
  }
}

const failures = [];
for (const [name, run] of CASES) {
  try {
    await runCaseWithWatchdog(run);
  } catch (error) {
    failures.push(`${name}: ${error?.message || error}`);
  }
}

if (failures.length > 0) {
  console.error(`${failures.length}/${CASES.length} bridge operation queue cases failed:`);
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exitCode = 1;
} else {
  console.log(`Bridge operation queue tests passed (${CASES.length} cases)`);
}
