// Reproducible in-process timings, independent from Python IPC/full generation.
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {cpus} from 'node:os';
import {performance} from 'node:perf_hooks';
import {parse} from '../src/ia_propia/assets/nerva/parser.mjs';
import {runProgram, VERSION} from '../src/ia_propia/assets/nerva/runtime.mjs';
import {prepareTokenSelector, DECODER_SHA256} from '../src/ia_propia/assets/nerva/token_selector.mjs';

const source = readFileSync(new URL('../src/ia_propia/assets/nerva/conversation_decoder.nv', import.meta.url), 'utf8');
const select = prepareTokenSelector(source);
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const results = [];
for (const size of [2048, 8192]) for (const temperature of [0, .7]) {
  const inputs = {scores: Array.from({length: size}, (_, i) => Math.sin(i * .31)), excluded: [0], generated: [1, 2, 2],
    temperature, top_k: temperature ? 40 : 0, top_p: temperature ? .9 : 1, repetition_penalty: 1.1, draw: .37};
  const payload = JSON.stringify(inputs);
  const expected = runProgram(source, {inputs}).records[0].values[1];
  assert.equal(select(inputs), expected);
  const functions = {
    parse_source: () => parse(source).body.length,
    parse_json: () => JSON.parse(payload).scores.length,
    interpreter: () => runProgram(source, {inputs}).records[0].values[1],
    specialized: () => select(inputs),
    json_interpreter: () => runProgram(source, {inputs: JSON.parse(payload)}).records[0].values[1],
    json_specialized: () => select(JSON.parse(payload)),
  };
  const names = Object.keys(functions), samples = Object.fromEntries(names.map(n => [n, []]));
  for (const fn of Object.values(functions)) for (let i = 0; i < 20; i++) fn();
  let checksum = 0;
  for (let round = 0; round < 5; round++) for (let offset = 0; offset < names.length; offset++) {
    const name = names[(round + offset) % names.length], fn = functions[name], start = performance.now();
    for (let call = 0; call < 60; call++) checksum += fn();
    samples[name].push((performance.now() - start) / 60);
  }
  assert.ok(Number.isFinite(checksum));
  results.push({vocabularySize: size, temperature, callsPerRound: 60, rounds: 5, parity: true,
    millisecondsPerDecision: Object.fromEntries(names.map(n => [n, {median: median(samples[n]), samples: samples[n]}]))});
}
const report = {schemaVersion: 1, scope: 'in_process_selector_not_complete_generation',
  runtimeVersion: VERSION, decoderSha256: DECODER_SHA256,
  environment: {node: process.version, cpu: cpus()[0]?.model, logicalCPUs: cpus().length}, results};
const text = JSON.stringify(report, null, 2) + '\n';
if (process.argv[2]) writeFileSync(process.argv[2], text, {flag: 'wx'});
process.stdout.write(text);
