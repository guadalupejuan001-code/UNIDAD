// Optional specialization of ONE pinned, data-only NERVA program.
// The public interpreter, parser, tensor primitives and original worker stay intact.
import { createHash } from 'node:crypto';
import { parse, LangError } from './parser.mjs';
import { runProgram } from './runtime.mjs';
import * as T from './tensor.mjs';

export const DECODER_SHA256 = '97d5b38f4e3aa0277eec0d37debbfdbba4e6b814eb281b1cbcd05fc9a2bfec4d';
const KEYS = ['scores', 'excluded', 'generated', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'draw'];

export function prepareTokenSelector(source) {
  // Parsing retains the language source bounds. The hash prevents a changed .nv
  // program from silently running old specialized semantics. No eval/codegen.
  const ast = parse(source);
  if (createHash('sha256').update(source).digest('hex') !== DECODER_SHA256)
    throw new LangError('Unsupported fixed decoder; use the public interpreter');
  // Every statement/expression of this straight-line program executes once.
  const count = node => {
    if (!node || typeof node !== 'object') return 0;
    if (Array.isArray(node)) return node.reduce((sum, item) => sum + count(item), 0);
    return (node.type && node.type !== 'program' ? 1 : 0) +
      Object.values(node).reduce((sum, item) => sum + count(item), 0);
  };
  const steps = count(ast);
  return function select(input, options = {}) {
    const operationLimit = options.operationLimit ?? 20000000;
    const stepLimit = options.stepLimit ?? 200000;
    if (!Number.isSafeInteger(operationLimit) || operationLimit < 1 || operationLimit > 20000000)
      throw new LangError('operationLimit debe estar entre 1 y 20000000.');
    if (!Number.isSafeInteger(stepLimit) || stepLimit < 1 || stepLimit > 200000)
      throw new LangError('stepLimit debe estar entre 1 y 200000.');
    if (!input || Array.isArray(input) || typeof input !== 'object' ||
        Object.keys(input).length !== KEYS.length || KEYS.some(key => !Object.hasOwn(input, key)))
      throw new LangError('Invalid decoder fields');
    // The fixed flat numeric schema is a strict subset of the interpreter input
    // limits: fewer than 74,000 numbers, depth 1, no strings or executable data.
    for (const [key, maximum] of [['scores', 65536], ['excluded', 4096], ['generated', 4096]]) {
      const values = input[key];
      if (!Array.isArray(values) || values.length > maximum || (key === 'scores' && !values.length))
        throw new LangError('Invalid bounded decoder array: ' + key);
      for (const value of values)
        if (typeof value !== 'number' || !Number.isFinite(value)) throw new LangError('Decoder numbers must be finite');
    }
    for (const key of KEYS.slice(3))
      if (typeof input[key] !== 'number' || !Number.isFinite(input[key])) throw new LangError('Decoder controls must be finite');
    // Reduced step budgets fail at precisely the interpreter's original step.
    // This path only throws; there is no backend fallback after a selector error.
    if (stepLimit < steps) return runProgram(source, { inputs: input, stepLimit, operationLimit }).records[0].values[1];
    T.resetBudget(operationLimit);
    // Keep tensor construction, float64 semantics, validation and operation
    // charging unchanged. In particular do not substitute the unbudgeted fusion
    // prototype or a JS/NumPy implementation of the sampling algorithm.
    const logits = T.tensor(input.scores, 'f64');
    const token = T.selectToken(logits, input.excluded, input.generated, input.temperature,
      input.top_k, input.top_p, input.repetition_penalty, input.draw);
    // Legacy global export charges serialize(logits) against the tensor budget.
    // Retain that bounded traversal so even reduced operation limits admit the
    // exact same valid decisions. Omit the expensive general global exporter,
    // its repeated number-to-string conversions and copies of all input lists.
    T.serialize(logits);
    return token;
    // Output is one integer <= 65535 (< 100 JSON characters). No globals are
    // exported by this separate API; runProgram/runSource2 keep their contract.
  };
}
