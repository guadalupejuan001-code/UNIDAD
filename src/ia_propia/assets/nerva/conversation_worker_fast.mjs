// Opt-in worker. Same JSON protocol, byte/decision budgets and lifetime as v0.2.
import { readFileSync } from 'node:fs';
import { prepareTokenSelector } from './token_selector.mjs';

const select = prepareTokenSelector(readFileSync(new URL('./conversation_decoder.nv', import.meta.url), 'utf8'));
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
      const token = select(JSON.parse(line));
      if (!Number.isSafeInteger(token)) throw new Error('Invalid decoder result');
      process.stdout.write(JSON.stringify({ token }) + '\n');
    }
  }
  if (pending) throw new Error('Incomplete decoder request');
} catch (error) {
  process.stdout.write(JSON.stringify({ error: String(error.message).slice(0, 1000) }) + '\n');
  process.exitCode = 1;
  process.stdin.destroy();
}
