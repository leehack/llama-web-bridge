// Tests of scripts/release/qualification.mjs's diagnostic redaction, one
// test per test method of scripts/release_qualification_test.py with the
// same name and assertions (see qualification_fixtures.mjs for the mapping),
// plus the Python regex semantics the port has to reproduce.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { PyFloat, pyJsonDumps, pyJsonLoads } from '../../scripts/release/json.mjs';
import {
  MAX_SANITIZE_NESTING,
  REDACTED_CREDENTIAL,
  normalizeTranscript,
  sanitizeDiagnosticStdout,
  sanitizeDiagnosticText,
  sanitizeDiagnosticValue,
} from '../../scripts/release/qualification.mjs';

// json.loads with the default hooks (NaN and Infinity accepted).
const loads = (text) => pyJsonLoads(text);
const dumps = (value) => pyJsonDumps(value);

function assertNotIn(needle, haystack) {
  assert.ok(!haystack.includes(needle), `${JSON.stringify(needle)} leaked into ${JSON.stringify(haystack)}`);
}

test('test_diagnostics_are_sanitized', () => {
  const raw = 'failed https://huggingface.co/model.gguf?token=secret123#frag and '
    + 'https://user:pass@example.com/a/b?x=1\n'
    + 'Authorization: Bearer ghp_this_must_not_escape\n'
    + 'GH_TOKEN=another-secret\n'
    + 'api_key=plain-secret';
  const sanitized = sanitizeDiagnosticText(raw);
  for (const leaked of ['secret123', 'token=', 'pass@', '#frag', '?x=1', 'ghp_this_must_not_escape', 'another-secret', 'plain-secret']) {
    assertNotIn(leaked, sanitized);
  }
  assert.ok(sanitized.includes('https://huggingface.co/model.gguf'));
  assert.ok(sanitized.includes('https://example.com/a/b'));
});

test('test_real_credentials_are_redacted_no_matter_where_they_appear', () => {
  const raw = 'Authorization: Basic dXNlcjpwYXNz\n'
    + 'apiKey: plain-secret\n'
    + 'MY_PASSWORD=hunter2\n'
    + 'service.credential = cred-value\n'
    + 'client_secret=shhh\n'
    + 'fetch https://example.com/m.gguf?sig=abc#frag';
  const sanitized = sanitizeDiagnosticText(raw);
  for (const leaked of ['dXNlcjpwYXNz', 'plain-secret', 'hunter2', 'cred-value', 'shhh', 'sig=abc', '#frag']) assertNotIn(leaked, sanitized);
  assert.ok(sanitized.includes('https://example.com/m.gguf'));
});

test('test_malformed_url_diagnostic_is_redacted_without_raising', () => {
  const sanitized = sanitizeDiagnosticText('failed https://user:pass@[broken?token=super-secret#frag');
  for (const leaked of ['user', 'pass', 'super-secret', 'token=', '#frag']) assertNotIn(leaked, sanitized);
  assert.ok(sanitized.includes('https://<redacted-url>'));
});

test('test_malformed_url_authorities_are_redacted_without_leaking', () => {
  for (const raw of [
    'https:///user:pass@example.com/model.gguf?token=secret#frag',
    'https://user:password/model.gguf?token=secret#frag',
    'HTTPS:///user:pass@example.com/model.gguf?token=secret#frag',
  ]) {
    assert.equal(sanitizeDiagnosticText(raw), 'https://<redacted-url>', raw);
  }
});

test('test_json_escaped_url_is_redacted_in_malformed_fallback', () => {
  const malformed = String.raw`{"modelUrl":"https:\/\/url-user:url-pass@example.com/model.gguf?sig=query-secret#fragment-secret"`;
  const serialized = sanitizeDiagnosticStdout(malformed);
  const decoded = loads(serialized);
  assert.equal(typeof decoded, 'string');
  for (const leaked of ['url-user', 'url-pass', 'query-secret', 'fragment-secret', 'sig=']) assertNotIn(leaked, serialized);
  assert.ok(decoded.includes('https://example.com/model.gguf'));
});

test('test_url_sanitization_is_case_insensitive', () => {
  const sanitized = sanitizeDiagnosticText('HTTPS://user:pass@example.com/m.gguf?sig=secret#frag');
  for (const leaked of ['user', 'pass', 'sig=secret', '#frag']) assertNotIn(leaked, sanitized);
  assert.equal(sanitized, 'https://example.com/m.gguf');
});

test('test_llama_token_counters_are_not_treated_as_credentials', () => {
  const raw = 'llama_context: n_tokens = 65\n'
    + 'llama_context: n_tokens_batch = 65\n'
    + 'decoded n_tokens=65 n_tokens_batch=65\n';
  assert.equal(sanitizeDiagnosticText(raw), raw);
});

test('test_plural_token_credentials_do_not_use_the_counter_carveout', () => {
  const raw = 'access_tokens=access-secret\n'
    + 'refresh_tokens_json="refresh secret"\n'
    + 'n_tokens=counter-shaped-secret\n'
    + 'n_tokens_secret=65\n';
  const sanitized = sanitizeDiagnosticText(raw);
  for (const leaked of ['access-secret', 'refresh secret', 'counter-shaped-secret']) assertNotIn(leaked, sanitized);
  assertNotIn('n_tokens_secret', sanitized);

  const structured = loads(sanitizeDiagnosticStdout(dumps({
    access_tokens: 'access-secret',
    n_tokens: 'counter-shaped-secret',
    n_tokens_batch: 65,
    n_tokens_api_key: 65,
  })));
  assert.equal(structured.access_tokens, REDACTED_CREDENTIAL);
  assert.equal(structured.n_tokens, REDACTED_CREDENTIAL);
  assert.equal(structured.n_tokens_batch, 65);
  assert.equal(structured.n_tokens_api_key, REDACTED_CREDENTIAL);
});

test('test_dotted_structured_credential_keys_are_redacted', () => {
  const structured = loads(sanitizeDiagnosticStdout(dumps({
    'service.credential': 'service-secret',
    'headers.apiKey': 'header-secret',
    'llama_context.n_tokens': 65,
    'headers.apiKey.n_tokens': 65,
  })));
  assert.equal(structured['service.credential'], REDACTED_CREDENTIAL);
  assert.equal(structured['headers.apiKey'], REDACTED_CREDENTIAL);
  assert.equal(structured['llama_context.n_tokens'], 65);
  assert.equal(structured['headers.apiKey.n_tokens'], REDACTED_CREDENTIAL);
});

test('test_credential_names_with_alphanumeric_suffixes_are_redacted', () => {
  const raw = 'tokenValue=token-secret\n'
    + 'secretValue=secret-secret\n'
    + 'passwordValue=password-secret\n'
    + 'credentialValue=credential-secret\n'
    + 'apiKeyV2=api-key-secret\n';
  const sanitized = sanitizeDiagnosticText(raw);
  for (const leaked of ['token-secret', 'secret-secret', 'password-secret', 'credential-secret', 'api-key-secret']) assertNotIn(leaked, sanitized);
});

test('test_quoted_credential_assignment_values_are_fully_redacted', () => {
  const raw = 'PASSWORD="correct horse battery staple"\n'
    + "api_key='quoted api key'\n";
  const sanitized = sanitizeDiagnosticText(raw);
  for (const leaked of ['correct', 'horse', 'battery', 'staple', 'quoted', 'api key']) assertNotIn(leaked, sanitized);
  assert.equal(sanitized, `${REDACTED_CREDENTIAL}\n${REDACTED_CREDENTIAL}\n`);
});

test('test_quoted_credential_keys_are_redacted_in_text_fallback', () => {
  const malformedJson = '{"apiKey": "plain secret", '
    + '"Authorization": "Bearer bearer-secret", '
    + '"password": "hunter2"';
  const sanitized = sanitizeDiagnosticStdout(malformedJson);
  assertNotIn('plain secret', sanitized);
  assertNotIn('bearer-secret', sanitized);
  assertNotIn('hunter2', sanitized);

  const counters = '"n_tokens": 65, "n_tokens_batch": 65';
  assert.equal(sanitizeDiagnosticText(counters), counters);
});

test('test_authorization_assignments_are_redacted_in_text_fallback', () => {
  for (const raw of [
    '{"Authorization":"opaque-secret"',
    'Authorization=Bearer bearer-secret',
    'authorization = Basic basic-secret',
  ]) {
    const serialized = sanitizeDiagnosticStdout(raw);
    assert.equal(typeof loads(serialized), 'string', raw);
    for (const leaked of ['opaque-secret', 'bearer-secret', 'basic-secret']) assertNotIn(leaked, serialized);
  }
});

test('test_non_scheme_authorization_value_is_fully_redacted', () => {
  const serialized = sanitizeDiagnosticStdout('Authorization: Digest username=alice, response=digest-secret');
  assert.equal(typeof loads(serialized), 'string');
  for (const leaked of ['alice', 'digest-secret', 'response=']) assertNotIn(leaked, serialized);
});

test('test_authorization_redaction_preserves_unrelated_diagnostics', () => {
  const diagnostics = 'authorizationStatus=enabled\n'
    + 'Basic validation completed\n'
    + 'Bearer capacity remains available';
  assert.equal(sanitizeDiagnosticText(diagnostics), diagnostics);

  const structured = { authorizationStatus: 'enabled', note: 'Basic validation completed' };
  assert.deepEqual(loads(sanitizeDiagnosticStdout(dumps(structured))), structured);
});

test('test_bearer_and_basic_credentials_are_redacted_in_all_fallbacks', () => {
  const malformed = 'Bearer bare-bearer-secret\n'
    + 'Basic bare-basic-secret\n'
    + 'access_token=Bearer assigned-bearer-secret';
  const serialized = sanitizeDiagnosticStdout(malformed);
  assert.equal(typeof loads(serialized), 'string');
  for (const leaked of ['bare-bearer-secret', 'bare-basic-secret', 'assigned-bearer-secret']) assertNotIn(leaked, serialized);

  const structured = sanitizeDiagnosticStdout(dumps({
    bearerLog: 'Bearer structured-bearer-secret',
    basicLog: 'Basic structured-basic-secret',
  }));
  for (const leaked of ['structured-bearer-secret', 'structured-basic-secret']) assertNotIn(leaked, structured);
});

test('test_nonstandard_or_nonfinite_stdout_uses_json_string_fallback', () => {
  const rejectNonstandardConstant = (value) => {
    throw new Error(`non-standard JSON constant: ${value}`);
  };
  for (const raw of [
    'NaN',
    'Infinity',
    '-Infinity',
    '1e9999',
    '{"apiKey":"must-not-escape","metric":NaN}',
    '{"apiKey":"must-not-escape","metric":1e9999}',
  ]) {
    const serialized = sanitizeDiagnosticStdout(raw);
    const decoded = pyJsonLoads(serialized, { parseConstant: rejectNonstandardConstant });
    assert.equal(typeof decoded, 'string', raw);
    assertNotIn('must-not-escape', serialized);
  }
});

test('test_structured_authorization_values_are_redacted', () => {
  const payload = { headers: { Authorization: 'Bearer bearer-secret', authorization: 'Basic basic-secret' } };
  assert.deepEqual(loads(sanitizeDiagnosticStdout(dumps(payload))), {
    headers: { Authorization: '<redacted-credential>', authorization: '<redacted-credential>' },
  });
});

test('test_flattened_authorization_names_are_redacted', () => {
  const payload = { 'headers.authorization': 'Bearer dotted-secret', HTTP_AUTHORIZATION: 'Basic environment-secret' };
  const serialized = sanitizeDiagnosticStdout(dumps(payload));
  assert.deepEqual(loads(serialized), { 'headers.authorization': REDACTED_CREDENTIAL, HTTP_AUTHORIZATION: REDACTED_CREDENTIAL });
  assertNotIn('dotted-secret', serialized);
  assertNotIn('environment-secret', serialized);

  const malformed = sanitizeDiagnosticStdout('HTTP_AUTHORIZATION=Bearer text-secret');
  assert.equal(typeof loads(malformed), 'string');
  assertNotIn('text-secret', malformed);
});

test('test_structured_keys_cannot_leak_urls_or_assignments', () => {
  const payload = {
    'https://user:pass@example.com/m.gguf?sig=secret#frag': 'url-key',
    'api_key=key-secret': 'assignment-key',
    ordinary: { n_tokens: 65 },
  };
  const serialized = sanitizeDiagnosticStdout(dumps(payload));
  const sanitized = loads(serialized);
  for (const leaked of ['user', 'pass', 'sig=secret', '#frag', 'key-secret']) assertNotIn(leaked, serialized);
  assert.equal(sanitized['https://example.com/m.gguf'], 'url-key');
  assert.equal(sanitized[REDACTED_CREDENTIAL], 'assignment-key');
  assert.equal(sanitized.ordinary.n_tokens, 65);
});

// --- Node-only: Python regex semantics the port reproduces -----------------------------

test('re.IGNORECASE matches the dotted and dotless i, long s and Kelvin sign', () => {
  // Python's re.IGNORECASE folds U+0130/U+0131 onto 'i', U+017F onto 's' and
  // U+212A onto 'k'; JavaScript's `i` flag misses the first two.
  assert.equal(sanitizeDiagnosticText('pİd=1 credentıal=2 CREDENTİAL=3'), `pİd=1 ${REDACTED_CREDENTIAL} ${REDACTED_CREDENTIAL}`);
  assert.equal(sanitizeDiagnosticText('ſecret=x'), REDACTED_CREDENTIAL);
  assert.equal(sanitizeDiagnosticText('apiKEY=x apı_key=y tOKen=z'), `${REDACTED_CREDENTIAL} ${REDACTED_CREDENTIAL} ${REDACTED_CREDENTIAL}`);
  assert.equal(sanitizeDiagnosticText('AUTHORİZATİON: x'), 'Authorization: <redacted>');
  assert.equal(sanitizeDiagnosticText('BEARER abc'), REDACTED_CREDENTIAL);
  // U+212A (Kelvin sign) is a 'k' in a literal and in [A-Za-z].
  assert.equal(sanitizeDiagnosticText('api\u212Aey=x \u212Atoken=y'), `${REDACTED_CREDENTIAL} ${REDACTED_CREDENTIAL}`);
  assert.equal(sanitizeDiagnosticText('n_tokens\u212A=1 n_tokens_\u212A=65'), `${REDACTED_CREDENTIAL} n_tokens_\u212A=65`);
  assert.deepEqual(sanitizeDiagnosticValue({ 'n_tokens_\u212A': 65, 'api\u212Aey': 1 }), { 'n_tokens_\u212A': 65, 'api\u212Aey': REDACTED_CREDENTIAL });
  assert.equal(sanitizeDiagnosticText('httpſ://u:p@h/x?q'), 'httpſ://<redacted-url>');
  assert.equal(sanitizeDiagnosticText('HtTpS:\\/\\/h.example:8080/a?b#c'), 'https://h.example:8080/a');
  assert.equal(normalizeTranscript('  LANGUAGE English <ASR_TEXT> Hello, World!'), 'hello world');
  assert.equal(normalizeTranscript('<aſr_text>Hi'), 'hi');
});

test('\\b, \\s and $ keep their Python meaning', () => {
  // Unicode letters are word characters, so there is no boundary inside 'étoken'.
  assert.equal(sanitizeDiagnosticText('étoken=1'), 'étoken=1');
  assert.equal(sanitizeDiagnosticText('é=1 xpassword=2'), `é=1 ${REDACTED_CREDENTIAL}`);
  // Python's \s includes \x1c-\x1f and \x85 and not U+FEFF.
  assert.equal(sanitizeDiagnosticText('token\x1c=\x85secret'), REDACTED_CREDENTIAL);
  assert.equal(sanitizeDiagnosticText('token\ufeff=secret'), 'token\ufeff=secret');
  // (?m)$ is only before '\n', and '.' is everything but '\n'.
  assert.equal(sanitizeDiagnosticText('Bearer abc\r\nnext'), `${REDACTED_CREDENTIAL}\r\nnext`);
  assert.equal(sanitizeDiagnosticText('Bearer abc\u2028next'), 'Bearer abc\u2028next');
  assert.equal(sanitizeDiagnosticText('token="a\\\u2028b"'), REDACTED_CREDENTIAL);
});

test('URL authorities follow urllib.parse', () => {
  const cases = [
    ['https://[::1]:443/p?q', 'https://[::1]:443/p'],
    ['https://u@[fe80::1%25eth0]/x', 'https://[fe80::1%25eth0]/x'],
    ['https://[1.2.3.4]/x', 'https://<redacted-url>'],
    ['https://[v1.x]/x', 'https://[v1.x]/x'],
    ['https://[vz.x]/x', 'https://<redacted-url>'],
    ['https://h:65536/x', 'https://<redacted-url>'],
    ['https://h:0065535/x', 'https://h:0065535/x'],
    ['https://h:/x', 'https://h:/x'],
    ['https://h:٣/x', 'https://<redacted-url>'],
    ['https://a@b@c/x', 'https://c/x'],
    ['https://ℂ/x', 'https://ℂ/x'],
    ['https://h℀x/y', 'https://<redacted-url>'],
    ['https://h#frag/x', 'https://h'],
  ];
  for (const [raw, expected] of cases) assert.equal(sanitizeDiagnosticText(raw), expected, raw);
});

test('structured sanitizing keeps the JSON value model and Python nesting bound', () => {
  assert.equal(
    sanitizeDiagnosticStdout('{"a": 1.0, "n_tokens": -1, "big": 123456789012345678901234567890, "k": "\\ud800"}'),
    `{"a": 1.0, "n_tokens": "${REDACTED_CREDENTIAL}", "big": 123456789012345678901234567890, "k": "\\ud800"}\n`,
  );
  // Duplicate keys: the last value at the first position, as json.loads keeps them.
  assert.equal(sanitizeDiagnosticStdout('{"a": 1, "b": 2, "a": 3}'), '{"a": 3, "b": 2}\n');
  // Sanitized keys that collide keep the first position and the last value.
  assert.equal(sanitizeDiagnosticStdout('{"token=1": 1, "b": 2, "secret=2": 3}'), `{"${REDACTED_CREDENTIAL}": 3, "b": 2}\n`);
  assert.equal(sanitizeDiagnosticValue(new PyFloat(2)).value, 2);
  const deep = (depth) => `${'['.repeat(depth)}1${']'.repeat(depth)}`;
  assert.equal(sanitizeDiagnosticStdout(deep(MAX_SANITIZE_NESTING)), `${deep(MAX_SANITIZE_NESTING)}\n`);
  assert.throws(() => sanitizeDiagnosticStdout(deep(MAX_SANITIZE_NESTING + 1)), (error) => error.pyType === 'RecursionError');
  // What a level's own work calls counts too (measured against Python 3.12):
  // an empty container is as deep as a scalar, a URL goes two frames further,
  // and an unredacted token counter's value is one level deeper than its dict.
  const nest = (depth, leaf) => `${'['.repeat(depth)}${leaf}${']'.repeat(depth)}`;
  assert.equal(sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING, '[]')), `${nest(MAX_SANITIZE_NESTING, '[]')}\n`);
  assert.throws(() => sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING + 1, '{}')), (error) => error.pyType === 'RecursionError');
  assert.equal(sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING - 2, '"https://h/x"')), `${nest(MAX_SANITIZE_NESTING - 2, '"https://h/x"')}\n`);
  assert.throws(() => sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING - 1, '"https://h/x"')), (error) => error.pyType === 'RecursionError');
  assert.equal(sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING, '{"token": 1}')), `${nest(MAX_SANITIZE_NESTING, `{"token": "${REDACTED_CREDENTIAL}"}`)}\n`);
  assert.throws(() => sanitizeDiagnosticStdout(nest(MAX_SANITIZE_NESTING, '{"n_tokens": 5}')), (error) => error.pyType === 'RecursionError');
});
