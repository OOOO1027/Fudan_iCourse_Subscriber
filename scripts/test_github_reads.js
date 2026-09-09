const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

async function run() {
  const calls = [];
  let responses = [];
  const ctx = { window: { ICS: {} }, fetch: async (url, options) => {
    calls.push({ url, options });
    return responses.shift();
  } };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(require('node:path').join(__dirname, '../frontend/js/github.js'), 'utf8'), ctx);
  const github = ctx.window.ICS.github;
  responses = [{ status: 401 }, { ok: true, json: async () => ({ object: { sha: 'abc' } }) }];
  assert.equal(await github.getLatestCommitSha('owner', 'repo', 'data', 'expired-test-token'), 'abc');
  assert.equal(calls.length, 2);
  assert.equal(calls[1].options.headers.Authorization, undefined);
  calls.length = 0;
  responses = [{ status: 403, ok: false, text: async () => 'rate limited' }];
  await assert.rejects(github.getLatestCommitSha('owner', 'repo', 'data', 'valid-test-token'), /403/);
  assert.equal(calls.length, 1);
  calls.length = 0;
  responses = [{ status: 401, ok: false, text: async () => 'Bad credentials' }];
  await assert.rejects(github.triggerSingleRunWorkflow('owner', 'repo', 'main', 'expired-test-token', ['123']), /401/);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.method, 'POST');
  console.log('GitHub expired-token read recovery and write isolation passed');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
