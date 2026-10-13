#!/usr/bin/env node
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { parse } from './parser.mjs';
import { runProgram, VERSION } from './runtime.mjs';

const HELP = `NERVA ${VERSION} — programación y aprendizaje local en CPU
Uso:
  node nerva.mjs run programa.nv [--json] [--seed 42] [--inputs datos.json] [--out carpeta]
  node nerva.mjs check programa.nv
  node nerva.mjs --version

--json devuelve {version,records,steps}; los errores salen por stderr.
--inputs define entradas constantes desde un objeto JSON.
--out guarda los checkpoints emitidos por el programa en esa carpeta.
Sin --out, el programa no escribe archivos. check valida solo la sintaxis.
Requiere Node.js 18+; no instala dependencias ni usa la red.
`;

function display(value) {
  if (value?.type === 'tensor') return `Tensor[${value.shape.join(',')}] ${value.dtype} ${JSON.stringify(value.data)}`;
  return typeof value === 'string' ? value : JSON.stringify(value);
}

async function main(args) {
  if (!args.length || args.includes('--help') || args.includes('-h')) { process.stdout.write(HELP); return; }
  if (args.length === 1 && args[0] === '--version') { process.stdout.write(`NERVA ${VERSION}\n`); return; }
  const [command, filename, ...rest] = args;
  if (!['run', 'check'].includes(command) || !filename || filename.startsWith('--')) throw new Error(HELP);
  const options = {}; let json = false, outputDirectory = null;
  for (let i = 0; i < rest.length; i++) {
    const name = rest[i];
    if (name === '--json' && command === 'run') { json = true; continue; }
    if (!['--seed', '--inputs', '--out'].includes(name) || command !== 'run') throw new Error(`Opción no reconocida: ${name}`);
    if (!rest[i + 1] || rest[i + 1].startsWith('--')) throw new Error(`Falta el valor de ${name}`);
    const value = rest[++i];
    if (name === '--seed') { if (!/^\d+$/.test(value)) throw new Error('La semilla debe ser un entero no negativo.'); options.seed = Number(value); }
    if (name === '--inputs') options.inputs = JSON.parse(await readFile(resolve(value), 'utf8'));
    if (name === '--out') outputDirectory = resolve(value);
  }
  const source = await readFile(resolve(filename), 'utf8');
  if (command === 'check') { parse(source); process.stdout.write('Sintaxis válida.\n'); return; }
  const result = runProgram(source, options);
  if (outputDirectory) {
    const checkpoints = result.records.filter(record => record.kind === 'checkpoint');
    if (checkpoints.length) await mkdir(outputDirectory, { recursive: true });
    for (const record of checkpoints) await writeFile(join(outputDirectory, `${record.name}.json`), JSON.stringify(record.checkpoint, null, 2) + '\n');
  }
  if (json) process.stdout.write(JSON.stringify({ version: VERSION, records: result.records, steps: result.steps }) + '\n');
  else for (const record of result.records) {
    if (record.kind === 'print') process.stdout.write(record.values.map(display).join(' ') + '\n');
    else process.stdout.write(`Checkpoint: ${record.name}${outputDirectory ? ' → ' + join(outputDirectory, record.name + '.json') : ' (usa --out para guardarlo)'}\n`);
  }
}

main(process.argv.slice(2)).catch(error => {
  process.stderr.write(JSON.stringify({ error: error.message, line: error.line ?? null, column: error.column ?? null }) + '\n');
  process.exitCode = 1;
});
