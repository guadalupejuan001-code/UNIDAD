// One bounded worker per reply, reused for all tokens. Only this fixed program
// runs; a request cannot supply source, modules, paths, commands or weights.
import { readFileSync } from 'node:fs';
import { runProgram } from './runtime.mjs';

const source = readFileSync(new URL('./conversation_decoder.nv', import.meta.url), 'utf8');
const keys = ['scores', 'excluded', 'generated', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'draw'];
let pending = '', count = 0;
process.stdin.setEncoding('utf8');
try {
  for await (const chunk of process.stdin) {
    pending += chunk;
    if (pending.length > 8 * 1024 * 1024) throw new Error('Decoder input exceeds its byte budget');
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, newline);
      pending = pending.slice(newline + 1);
      if (++count > 1024) throw new Error('Decoder exceeds 1024 decisions');
      const input = JSON.parse(line);
      if (!input || Array.isArray(input) || Object.keys(input).length !== keys.length || keys.some(k => !Object.hasOwn(input, k)))
        throw new Error('Invalid decoder fields');
      const result = runProgram(source, { inputs: input });
      const record = result.records[0];
      if (result.records.length !== 1 || record?.values?.[0] !== 'token' || !Number.isSafeInteger(record.values[1]))
        throw new Error('Invalid decoder result');
      process.stdout.write(JSON.stringify({ token: record.values[1] }) + '\n');
    }
  }
  if (pending) throw new Error('Incomplete decoder request');
} catch (error) {
  process.stdout.write(JSON.stringify({ error: String(error.message).slice(0, 1000) }) + '\n');
  process.exitCode = 1;
  process.stdin.destroy();
}
