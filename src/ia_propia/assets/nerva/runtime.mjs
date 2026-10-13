// NERVA 0.2: a bounded interpreter, not JavaScript evaluation.
import { parse, LangError } from './parser.mjs';
import * as T from './tensor.mjs';

export const VERSION = '0.2.0';
const MAX_STEPS = 200000;
const MAX_OUTPUT = 1000;
const MAX_OUTPUT_CHARS = 10000000;
const MAX_LIST = 100000;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

class Environment {
  constructor(parent = null) { this.parent = parent; this.bindings = new Map(); }
  define(name, value, mutable = false) {
    if (this.bindings.has(name)) throw new Error(`Ya existe '${name}' en este ámbito.`);
    this.bindings.set(name, { value, mutable });
  }
  resolve(name) {
    if (this.bindings.has(name)) return this.bindings.get(name);
    if (this.parent) return this.parent.resolve(name);
    throw new Error(`Nombre desconocido: '${name}'.`);
  }
  assign(name, value) {
    const binding = this.resolve(name);
    if (!binding.mutable) throw new Error(`'${name}' es constante; no puede reasignarse.`);
    binding.value = value;
  }
}
class Builtin {
  constructor(name, fn) { this.name = name; this.fn = fn; }
}
class UserFunction {
  constructor(node, env) { this.node = node; this.env = env; }
}
class Signal {
  constructor(kind, value = null) { this.kind = kind; this.value = value; }
}

function finite(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Se esperaba un número finito.');
  return value;
}
function integer(value, name = 'valor') {
  finite(value);
  if (!Number.isSafeInteger(value)) throw new Error(`${name} debe ser un entero seguro.`);
  return value;
}
function boolean(value) {
  if (typeof value !== 'boolean') throw new Error('La condición debe ser booleana; compara valores explícitamente.');
  return value;
}
function asTensor(value) {
  if (!(value instanceof T.Tensor)) throw new Error('Se esperaba un Tensor. Usa tensor([...]).');
  return value;
}
function shapeEqual(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }
function params(value) {
  const result = Array.isArray(value) ? value : [value];
  if (!result.length || result.length > 32) throw new Error('Se requieren entre 1 y 32 parámetros.');
  if (new Set(result).size !== result.length) throw new Error('Un parámetro no puede aparecer dos veces.');
  for (const p of result) {
    asTensor(p);
    if (!p.requiresGrad || (p._parents && p._parents.length)) throw new Error('Se requieren parámetros hoja declarados con param.');
  }
  return result;
}
function scalar(value) {
  if (value instanceof T.Tensor) {
    if (value.shape.length !== 0) throw new Error('value() necesita un Tensor escalar.');
    return finite(value.data[0]);
  }
  return finite(value);
}
function plain(value, depth = 0, reserve = () => {}, describeFunctions = false) {
  reserve(1, 2);
  if (depth > 64) throw new Error('La salida supera 64 niveles.');
  if (value instanceof T.Tensor) {
    let prefix = 1, count = 0;
    for (const size of value.shape) { prefix *= size; count += prefix; }
    reserve(count, count * 2 + 100);
    for (const number of value.data) reserve(0, String(finite(number)).length);
    return T.serialize(value);
  }
  if (Array.isArray(value)) { reserve(0, value.length); return value.map(v => plain(v, depth + 1, reserve, describeFunctions)); }
  if (describeFunctions && (value instanceof UserFunction || value instanceof Builtin)) {
    const name = value.node?.name ?? value.name; reserve(0, name.length + 40);
    return { type: 'function', name };
  }
  if (value === null || typeof value === 'boolean' || typeof value === 'string') { reserve(0, JSON.stringify(value).length); return value; }
  if (typeof value === 'number') { reserve(0, String(finite(value)).length); return value; }
  throw new Error('Este valor no se puede imprimir ni exportar.');
}

export function runProgram(source, options = {}) {
  const ast = parse(source);
  const operationLimit = options.operationLimit ?? 20000000;
  if (!Number.isSafeInteger(operationLimit) || operationLimit < 1 || operationLimit > 20000000) throw new LangError('operationLimit debe estar entre 1 y 20000000.');
  T.resetBudget(operationLimit);
  const limit = options.stepLimit ?? MAX_STEPS;
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > MAX_STEPS) throw new LangError('stepLimit debe estar entre 1 y 200000.');
  let steps = 0, outputChars = 0, serializationNodes = 0, serializationChars = 0, depth = 0, loopDepth = 0;
  let randomState = 42;
  const records = [];
  const base = new Environment();
  const global = new Environment(base);
  const tick = node => {
    steps += 1;
    if (steps > limit) throw new LangError(`Se agotó el límite de ${limit} pasos.`, { line: node?.line, column: node?.column });
  };
  const record = value => {
    if (records.length >= MAX_OUTPUT) throw new Error('La ejecución supera 1000 salidas.');
    const bytes = JSON.stringify(value).length;
    if (outputChars + bytes > MAX_OUTPUT_CHARS) throw new Error('La salida supera 10 millones de caracteres.');
    outputChars += bytes; records.push(value);
  };
  const reserve = (count, chars = 0) => {
    if (!Number.isSafeInteger(count) || count < 0 || serializationNodes + count > 1000000) throw new Error('La serialización supera el límite de 1000000 valores y contenedores.');
    if (!Number.isSafeInteger(chars) || chars < 0 || serializationChars + chars > MAX_OUTPUT_CHARS) throw new Error('La serialización supera el límite de 10 millones de caracteres.');
    serializationNodes += count;
    serializationChars += chars;
  };
  const builtin = (name, min, max, fn) => base.define(name, new Builtin(name, args => {
    if (args.length < min || args.length > max) throw new Error(`${name} espera ${min === max ? min : `${min}–${max}`} argumentos; recibió ${args.length}.`);
    return fn(...args);
  }));
  const seed = value => {
    integer(value, 'La semilla');
    if (value < 0 || value > 4294967295) throw new Error('La semilla debe estar entre 0 y 4294967295.');
    randomState = value >>> 0; return null;
  };
  const uniform = () => {
    randomState = (randomState + 0x6D2B79F5) >>> 0;
    let z = randomState;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  seed(options.seed ?? 42);
  builtin('seed', 1, 1, seed);
  builtin('tensor', 1, 1, value => T.tensor(value, 'f32'));
  builtin('tensor64', 1, 1, value => T.tensor(value, 'f64'));
  builtin('zeros', 1, 2, (shape, dtype = 'f32') => T.zeros(shape, dtype));
  builtin('ones', 1, 2, (shape, dtype = 'f32') => T.ones(shape, dtype));
  builtin('randn', 1, 2, (shape, scale = 1) => {
    finite(scale); const result = T.zeros(shape);
    for (let i = 0; i < result.data.length; i++) {
      const u = Math.max(uniform(), Number.MIN_VALUE), v = uniform();
      const sample = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
      result.data[i] = sample;
      if (!Number.isFinite(result.data[i])) throw new Error('randn produjo un valor fuera del rango del Tensor.');
    }
    return result;
  });
  builtin('shape', 1, 1, value => [...asTensor(value).shape]);
  builtin('reshape', 2, 2, (value, shape) => T.reshape(asTensor(value), shape));
  builtin('transpose', 1, 1, value => T.transpose(asTensor(value)));
  for (const name of ['relu', 'tanh', 'sigmoid', 'exp', 'log', 'sum', 'mean', 'softmax']) builtin(name, 1, 1, value => T[name](asTensor(value)));
  builtin('mse', 2, 2, (a, b) => T.mse(asTensor(a), asTensor(b)));
  builtin('cross_entropy', 2, 2, (logits, labels) => T.crossEntropy(asTensor(logits), labels));
  builtin('argmax', 1, 1, value => T.argmax(asTensor(value)));
  builtin('detach', 1, 1, value => T.detach(asTensor(value)));
  builtin('backward', 1, 1, loss => { T.backward(asTensor(loss)); return null; });
  builtin('zero_grad', 1, 1, value => { T.zeroGrad(params(value)); return null; });
  builtin('step', 2, 2, (value, lr) => { T.step(params(value), finite(lr)); return null; });
  builtin('grad', 1, 1, value => {
    const p = asTensor(value);
    if (p.grad === null) throw new Error('Este Tensor aún no tiene gradiente; llama backward(loss).');
    return T.fromFlat(p.grad, p.shape, 'f64');
  });
  builtin('value', 1, 1, scalar);
  builtin('print', 1, 32, (...values) => { record({ kind: 'print', values: values.map(v => plain(v, 0, reserve)), version: '0.2' }); return null; });
  builtin('assert', 1, 2, (test, message = 'La comprobación falló.') => {
    if (!boolean(test)) throw new Error(typeof message === 'string' ? message : 'La comprobación falló.');
    return null;
  });
  builtin('range', 1, 3, (a, b = undefined, stride = 1) => {
    let start = b === undefined ? 0 : integer(a), end = b === undefined ? integer(a) : integer(b);
    integer(stride, 'El paso');
    if (!stride) throw new Error('range no admite paso cero.');
    const count = Math.max(0, Math.ceil((end - start) / stride));
    if (count > MAX_LIST) throw new Error('range supera 100000 elementos.');
    return Array.from({ length: count }, (_, i) => start + i * stride);
  });
  builtin('len', 1, 1, value => {
    if (Array.isArray(value)) return value.length;
    if (typeof value === 'string') return Array.from(value).length;
    if (value instanceof T.Tensor && value.shape.length) return value.shape[0];
    throw new Error('len espera una lista, una cadena o un Tensor con dimensiones.');
  });
  builtin('abs', 1, 1, value => Math.abs(finite(value)));
  builtin('sqrt', 1, 1, value => value instanceof T.Tensor ? T.pow(value, 0.5) : finite(Math.sqrt(finite(value))));
  builtin('min', 2, 2, (a, b) => Math.min(finite(a), finite(b)));
  builtin('max', 2, 2, (a, b) => Math.max(finite(a), finite(b)));
  builtin('checkpoint', 2, 2, (name, values) => {
    if (typeof name !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error('El nombre del checkpoint admite 1–64 letras ASCII, números, _ o -.');
    const checkpoint = { format: 'nerva-checkpoint', version: 1, tensors: params(values).map(p => {
      const { shape, dtype, data } = plain(p, 0, reserve); return { shape, dtype, data };
    }) };
    record({ kind: 'checkpoint', name, checkpoint, version: '0.2' });
    return JSON.stringify(checkpoint);
  });
  builtin('restore', 2, 2, (text, values) => {
    if (typeof text !== 'string' || text.length > MAX_OUTPUT_CHARS) throw new Error('restore espera una cadena JSON de checkpoint de hasta 10 millones de caracteres.');
    const ps = params(values);
    let data;
    try { data = JSON.parse(text); } catch { throw new Error('El checkpoint no es JSON válido.'); }
    if (!data || data.format !== 'nerva-checkpoint' || data.version !== 1 || !Array.isArray(data.tensors) || data.tensors.length !== ps.length) throw new Error('Formato o número de parámetros del checkpoint incorrecto.');
    const prepared = data.tensors.map((p, i) => {
      if (!p || !Array.isArray(p.shape) || !shapeEqual(p.shape, ps[i].shape) || p.dtype !== ps[i].dtype) throw new Error(`El parámetro ${i} del checkpoint tiene dimensiones o dtype incompatibles.`);
      const tensor = T.tensor(p.data, p.dtype);
      if (!shapeEqual(tensor.shape, p.shape)) throw new Error(`Los datos del parámetro ${i} no coinciden con sus dimensiones.`);
      return tensor;
    });
    // Validate every parameter before any change, so a bad checkpoint is atomic.
    ps.forEach((p, i) => { p.data.set(prepared[i].data); p.grad = null; p._version++; });
    return null;
  });

  if (options.inputs !== undefined) {
    if (!options.inputs || Array.isArray(options.inputs) || typeof options.inputs !== 'object') throw new LangError('inputs debe ser un objeto JSON de nombres y valores.');
    let inputNodes = 0, inputChars = 0;
    for (const [name, value] of Object.entries(options.inputs)) {
      if (!NAME.test(name) || base.bindings.has(name)) throw new LangError(`Nombre de entrada no permitido: '${name}'.`);
      // A host may pass only plain JSON values; objects and functions never cross.
      const validate = (v, d = 0) => {
        if (++inputNodes > 1000000) throw new LangError('La entrada supera 1000000 valores y contenedores.');
        if (d > 64) throw new LangError('La entrada supera 64 niveles.');
        if (Array.isArray(v)) { if (v.length > MAX_LIST) throw new LangError('Lista de entrada demasiado grande.'); v.forEach(x => validate(x, d + 1)); }
        else if (typeof v === 'number') finite(v);
        else if (typeof v === 'string') { inputChars += v.length; if (v.length > 200000 || inputChars > MAX_OUTPUT_CHARS) throw new LangError('Las cadenas de entrada superan el límite de tamaño.'); }
        else if (v !== null && !['string', 'boolean'].includes(typeof v)) throw new LangError('Las entradas admiten números, cadenas, booleanos, null y listas.');
      };
      validate(value); global.define(name, value);
    }
  }
  const wrap = (node, fn) => {
    try { return fn(); }
    catch (error) {
      if (error instanceof Signal || error instanceof LangError) throw error;
      throw new LangError(error?.message || String(error), { line: node?.line, column: node?.column });
    }
  };
  function evaluate(node, env) {
    tick(node);
    return wrap(node, () => {
      switch (node.type) {
        case 'literal': return node.value;
        case 'id': return env.resolve(node.name).value;
        case 'list': return node.items.map(item => evaluate(item, env));
        case 'unary': {
          const arg = evaluate(node.arg, env);
          if (node.op === 'not') return !boolean(arg);
          if (node.op === '+') return arg instanceof T.Tensor ? arg : finite(arg);
          if (node.op === '-') return arg instanceof T.Tensor ? T.neg(arg) : -finite(arg);
          throw new Error(`Operador desconocido: ${node.op}.`);
        }
        case 'binary': {
          const a = evaluate(node.left, env);
          if (node.op === 'and') return boolean(a) && boolean(evaluate(node.right, env));
          if (node.op === 'or') return boolean(a) || boolean(evaluate(node.right, env));
          const b = evaluate(node.right, env), tensor = a instanceof T.Tensor || b instanceof T.Tensor;
          if (node.op === '@') return T.matmul(asTensor(a), asTensor(b));
          const operators = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '^': 'pow' };
          if (operators[node.op] && tensor) return T[operators[node.op]](a, b);
          if (node.op === '+' && typeof a === 'string' && typeof b === 'string') {
            if (a.length + b.length > 200000) throw new Error('La cadena supera 200000 caracteres.');
            return a + b;
          }
          if (node.op === '==' || node.op === '!=') {
            if (tensor || Array.isArray(a) || Array.isArray(b) || typeof a === 'object' && a !== null || typeof b === 'object' && b !== null) throw new Error('Compara valores escalares; para Tensor escalar usa value().');
            return node.op === '==' ? a === b : a !== b;
          }
          finite(a); finite(b);
          switch (node.op) {
            case '+': return finite(a + b); case '-': return finite(a - b);
            case '*': return finite(a * b); case '/': return finite(a / b);
            case '^': return finite(a ** b);
            case '<': return a < b; case '<=': return a <= b;
            case '>': return a > b; case '>=': return a >= b;
            default: throw new Error(`Operador desconocido: ${node.op}.`);
          }
        }
        case 'index': {
          const target = evaluate(node.target, env), index = integer(evaluate(node.index, env), 'El índice');
          const values = typeof target === 'string' ? Array.from(target) : target;
          if (values instanceof T.Tensor) return T.sliceFirst(values, index);
          if (!Array.isArray(values) || index < 0 || index >= values.length) throw new Error('Índice fuera de rango o valor no indexable.');
          return values[index];
        }
        case 'call': {
          const callee = evaluate(node.callee, env);
          if (node.args.length > 32) throw new Error('Una llamada admite hasta 32 argumentos.');
          const args = node.args.map(arg => evaluate(arg, env));
          if (callee instanceof Builtin) return callee.fn(args);
          if (!(callee instanceof UserFunction)) throw new Error('Este valor no es una función.');
          if (args.length !== callee.node.params.length) throw new Error(`${callee.node.name} espera ${callee.node.params.length} argumentos.`);
          if (depth >= 64) throw new Error('La llamada supera el límite de 64 niveles de funciones.');
          const local = new Environment(callee.env);
          callee.node.params.forEach((name, i) => local.define(name, args[i], true));
          const oldLoopDepth = loopDepth; loopDepth = 0; depth++;
          try { executeBlock(callee.node.body, local); return null; }
          catch (signal) { if (signal instanceof Signal && signal.kind === 'return') return signal.value; throw signal; }
          finally { depth--; loopDepth = oldLoopDepth; }
        }
        default: throw new Error(`Expresión desconocida: ${node.type}.`);
      }
    });
  }
  function execute(node, env) {
    tick(node);
    return wrap(node, () => {
      switch (node.type) {
        case 'let': {
          let value = evaluate(node.expr, env);
          if (node.kind === 'param') { value = asTensor(value); value = T.tensor(value, value.dtype, true); }
          if (node.shape && !shapeEqual(asTensor(value).shape, node.shape)) throw new Error(`'${node.name}' requiere Tensor[${node.shape.join(',')}]; recibió Tensor[${value.shape.join(',')}].`);
          env.define(node.name, value, node.kind === 'let'); return;
        }
        case 'fn': env.define(node.name, new UserFunction(node, env)); return;
        case 'assign': env.assign(node.name, evaluate(node.expr, env)); return;
        case 'expr': evaluate(node.expr, env); return;
        case 'return':
          if (!depth) throw new Error('return solo se permite dentro de una función.');
          throw new Signal('return', node.expr ? evaluate(node.expr, env) : null);
        case 'break': case 'continue':
          if (!loopDepth) throw new Error(`${node.type} solo se permite dentro de un bucle.`);
          throw new Signal(node.type);
        case 'if': executeBlock(boolean(evaluate(node.test, env)) ? node.then : node.else, new Environment(env)); return;
        case 'for': case 'while': {
          let iterable = null;
          if (node.type === 'for') {
            iterable = evaluate(node.iterable, env);
            if (!Array.isArray(iterable)) throw new Error('for espera una lista; utiliza range().');
          }
          let i = 0; loopDepth++;
          try {
            while (node.type === 'for' ? i < iterable.length : boolean(evaluate(node.test, env))) {
              if (i >= MAX_LIST) throw new Error('El bucle supera 100000 iteraciones.');
              tick(node);
              const local = new Environment(env);
              if (iterable) local.define(node.name, iterable[i]);
              i++;
              try { executeBlock(node.body, local); }
              catch (signal) {
                if (signal instanceof Signal && signal.kind === 'break') break;
                if (signal instanceof Signal && signal.kind === 'continue') continue;
                throw signal;
              }
            }
          } finally { loopDepth--; }
          return;
        }
        default: throw new Error(`Instrucción desconocida: ${node.type}.`);
      }
    });
  }
  function executeBlock(body, env) { for (const node of body) execute(node, env); }
  executeBlock(ast.body, global);
  const globals = Object.create(null);
  for (const [name, binding] of global.bindings) {
    if (binding.value instanceof UserFunction || binding.value instanceof Builtin) continue;
    globals[name] = wrap(ast.body.at(-1), () => { reserve(0, JSON.stringify(name).length + 2); return plain(binding.value, 0, reserve, true); });
  }
  return { records, globals, steps };
}

export function runSource2(source, options = {}) { return runProgram(source, options).records; }
