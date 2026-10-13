/* NERVA 0.2: bounded CPU tensors and reverse-mode differentiation. */
const MAX_ELEMENTS = 1_000_000;
const MAX_RANK = 8;
const DEFAULT_BUDGET = 20_000_000;
let budgetLimit = DEFAULT_BUDGET;
let budgetUsed = 0;

export function resetBudget(limit = DEFAULT_BUDGET) {
  if (!Number.isSafeInteger(limit) || limit < 0) throw new RangeError('Operation budget must be a nonnegative safe integer');
  budgetLimit = limit;
  budgetUsed = 0;
  return getBudget();
}
export const setBudget = resetBudget;
export function getBudget() { return { limit: budgetLimit, used: budgetUsed, remaining: budgetLimit - budgetUsed }; }
function charge(count) {
  if (!Number.isSafeInteger(count) || count < 0 || count > budgetLimit - budgetUsed) throw new RangeError('Tensor operation budget exceeded');
  budgetUsed += count;
}

// Inference-only token selection. No tensor/parameter is mutated and there is
// no vocabulary text, answer template, filesystem or external model here.
export function selectToken(logits, excluded, generated, temperature, topK, topP, penalty, draw) {
  if (!(logits instanceof Tensor) || logits.shape.length !== 1 || !logits.data.length || logits.data.length > 65536)
    throw new RangeError('select_token requires a vector of 1..65536 logits');
  const size = logits.data.length;
  for (const ids of [excluded, generated]) {
    if (!Array.isArray(ids) || ids.length > 4096 || ids.some(id => !Number.isSafeInteger(id) || id < 0 || id >= size))
      throw new RangeError('select_token requires bounded valid token IDs');
  }
  for (const [value, low, high] of [[temperature, 0, 2], [topK, 0, 65536], [topP, 0.000001, 1], [penalty, 1, 2], [draw, 0, 1]]) {
    finite(value);
    if (value < low || value > high) throw new RangeError('Invalid select_token control');
  }
  if (!Number.isSafeInteger(topK) || draw === 1) throw new RangeError('Invalid top_k or random draw');
  charge(size * (8 + Math.ceil(Math.log2(size + 1))) + excluded.length + generated.length);
  const scores = Array.from(logits.data, x => finite(x));
  for (const id of new Set(generated)) scores[id] = scores[id] < 0 ? scores[id] * penalty : scores[id] / penalty;
  for (const id of excluded) scores[id] = -Infinity;
  let best = -1;
  for (let id = 0; id < size; id++) if (scores[id] !== -Infinity && (best < 0 || scores[id] > scores[best])) best = id;
  if (best < 0) throw new RangeError('All tokens are excluded');
  if (temperature === 0) return best;
  let order = scores.map((_, id) => id).filter(id => scores[id] !== -Infinity);
  order.sort((a, b) => scores[b] - scores[a] || a - b);
  if (topK) order = order.slice(0, topK);
  const maximum = scores[order[0]];
  // Subtract before division: small positive temperatures remain stable.
  const weights = order.map(id => Math.exp((scores[id] - maximum) / temperature));
  let total = weights.reduce((a, b) => a + b, 0);
  if (topP < 1) {
    let cumulative = 0, keep = 0;
    do { cumulative += weights[keep++]; } while (keep < weights.length && cumulative < topP * total);
    order = order.slice(0, keep);
    weights.length = keep;
    total = cumulative;
  }
  let cursor = draw * total;
  for (let i = 0; i < order.length; i++) {
    cursor -= weights[i];
    if (cursor < 0) return order[i];
  }
  return order[order.length - 1];
}
function finite(x, what = 'Tensor value') {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new RangeError(`${what} must be finite`);
  return x;
}
function checkDtype(dtype) {
  if (dtype !== 'f32' && dtype !== 'f64') throw new TypeError('Tensor dtype must be f32 or f64');
  return dtype;
}
function checkShape(shape) {
  if (!Array.isArray(shape) || shape.length > MAX_RANK) throw new RangeError(`Tensor shape must be an array of rank at most ${MAX_RANK}`);
  let count = 1;
  for (const dim of shape) {
    if (!Number.isSafeInteger(dim) || dim < 0 || dim > MAX_ELEMENTS) throw new RangeError('Tensor dimensions must be nonnegative integers at most 1000000');
    count *= dim;
    if (count > MAX_ELEMENTS) throw new RangeError(`Tensor may have at most ${MAX_ELEMENTS} elements`);
  }
  return count;
}
function createArray(data, dtype, count) {
  if (!Array.isArray(data) && !ArrayBuffer.isView(data)) throw new TypeError('Flat tensor data must be an array');
  if (data.length !== count) throw new RangeError('Tensor data length does not match shape');
  charge(count);
  const out = dtype === 'f32' ? new Float32Array(count) : new Float64Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = finite(data[i]);
    finite(out[i], `Value at index ${i} after ${dtype} conversion`);
  }
  return out;
}

export class Tensor {
  constructor(data, shape, dtype = 'f32', requiresGrad = false) {
    const count = checkShape(shape);
    this.dtype = checkDtype(dtype);
    this.data = createArray(data, dtype, count);
    this.shape = [...shape];
    this.requiresGrad = Boolean(requiresGrad);
    this.grad = null;
    this._parents = [];
    this._versions = [];
    this._backward = null;
    this._version = 0;
  }
}
function nestedData(value) {
  let leaves = 0;
  function visit(v, depth) {
    charge(1);
    if (typeof v === 'number') {
      if (++leaves > MAX_ELEMENTS) throw new RangeError(`Tensor may have at most ${MAX_ELEMENTS} elements`);
      return { shape: [], flat: [finite(v)] };
    }
    if (!Array.isArray(v) && !ArrayBuffer.isView(v)) throw new TypeError('Tensor input must contain numbers or rectangular lists');
    if (depth >= MAX_RANK) throw new RangeError(`Tensor rank may not exceed ${MAX_RANK}`);
    if (v.length > MAX_ELEMENTS) throw new RangeError('Tensor dimension exceeds 1000000');
    if (!v.length) return { shape: [0], flat: [] };
    const first = visit(v[0], depth + 1);
    const shape = [v.length, ...first.shape];
    checkShape(shape);
    const flat = [...first.flat];
    for (let i = 1; i < v.length; i++) {
      const child = visit(v[i], depth + 1);
      if (!sameShape(child.shape, first.shape)) throw new RangeError('Tensor lists must be rectangular');
      for (const x of child.flat) flat.push(x);
    }
    return { shape, flat };
  }
  return visit(value, 0);
}
export function tensor(data, dtype = undefined, requiresGrad = false) {
  if (data instanceof Tensor) return fromFlat(data.data, data.shape, dtype ?? data.dtype, requiresGrad);
  const parsed = nestedData(data);
  return fromFlat(parsed.flat, parsed.shape, dtype ?? 'f32', requiresGrad);
}
export function fromFlat(data, shape, dtype = 'f32', requiresGrad = false) { return new Tensor(data, shape, dtype, requiresGrad); }
export function zeros(shape, dtype = 'f32') {
  const count = checkShape(shape);
  charge(count);
  return fromFlat(new Float64Array(count), shape, dtype);
}
export function ones(shape, dtype = 'f32') {
  const count = checkShape(shape);
  charge(count);
  return fromFlat(new Float64Array(count).fill(1), shape, dtype);
}
function sameShape(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }
function requireTensor(t) { if (!(t instanceof Tensor)) throw new TypeError('Expected a tensor'); return t; }
function asTensor(v, dtype = 'f32') { return v instanceof Tensor ? v : tensor(v, dtype); }
function pair(a, b) {
  const dtype = (a instanceof Tensor ? a.dtype : b instanceof Tensor ? b.dtype : 'f32');
  return [asTensor(a, dtype), asTensor(b, dtype)];
}
function result(data, shape, dtype, parents = [], backwardFn = null) {
  const t = fromFlat(data, shape, dtype, parents.some(p => p.requiresGrad));
  if (t.requiresGrad) {
    t._parents = parents;
    t._versions = parents.map(p => p._version);
    t._backward = backwardFn;
  }
  return t;
}
function strides(shape) {
  const out = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) { out[i] = s; s *= shape[i]; }
  return out;
}
function broadcast(a, b) {
  const rank = Math.max(a.length, b.length);
  const shape = new Array(rank);
  for (let i = 0; i < rank; i++) {
    const x = a[a.length - 1 - i] ?? 1;
    const y = b[b.length - 1 - i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new RangeError(`Cannot broadcast shapes [${a}] and [${b}]`);
    shape[rank - 1 - i] = x === 1 ? y : x;
  }
  checkShape(shape);
  return shape;
}
function broadcastStrides(input, output) {
  const s = strides(input);
  return output.map((_, i) => {
    const j = i - (output.length - input.length);
    return j < 0 || input[j] === 1 ? 0 : s[j];
  });
}
function offsets(flat, shape, s, a, b) {
  let ai = 0, bi = 0;
  for (let d = 0; d < shape.length; d++) {
    const coordinate = Math.floor(flat / s[d]) % shape[d];
    ai += coordinate * a[d];
    bi += coordinate * b[d];
  }
  return [ai, bi];
}
function elementwise(a, b, calculate, derivative, name) {
  [a, b] = pair(a, b);
  const shape = broadcast(a.shape, b.shape);
  const count = checkShape(shape);
  const dtype = a.dtype === 'f64' || b.dtype === 'f64' ? 'f64' : 'f32';
  const os = strides(shape), sa = broadcastStrides(a.shape, shape), sb = broadcastStrides(b.shape, shape);
  charge(count * (Math.max(1, shape.length) + 1));
  const data = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const [ai, bi] = offsets(i, shape, os, sa, sb);
    data[i] = finite(calculate(a.data[ai], b.data[bi]), `${name} result`);
  }
  return result(data, shape, dtype, [a, b], (g, accumulate) => {
    charge(count * (Math.max(1, shape.length) + 4));
    const ga = a.requiresGrad ? new Float64Array(a.data.length) : null;
    const gb = b.requiresGrad ? new Float64Array(b.data.length) : null;
    for (let i = 0; i < count; i++) {
      const [ai, bi] = offsets(i, shape, os, sa, sb);
      const [da, db] = derivative(a.data[ai], b.data[bi], data[i], Boolean(ga), Boolean(gb));
      if (ga) ga[ai] = finite(ga[ai] + g[i] * finite(da, `${name} derivative`), 'Gradient');
      if (gb) gb[bi] = finite(gb[bi] + g[i] * finite(db, `${name} derivative`), 'Gradient');
    }
    if (ga) accumulate(a, ga);
    if (gb) accumulate(b, gb);
  });
}
export const add = (a, b) => elementwise(a, b, (x, y) => x + y, () => [1, 1], 'add');
export const sub = (a, b) => elementwise(a, b, (x, y) => x - y, () => [1, -1], 'subtract');
export const mul = (a, b) => elementwise(a, b, (x, y) => x * y, (x, y) => [y, x], 'multiply');
export const div = (a, b) => elementwise(a, b, (x, y) => x / y, (x, y) => [1 / y, -(x / y) / y], 'divide');
export const pow = (a, b) => elementwise(a, b, (x, y) => x ** y, (x, y, z, needA, needB) => [
  needA ? y === 0 ? 0 : y * x ** (y - 1) : 0,
  needB ? x > 0 ? z * Math.log(x) : NaN : 0,
], 'power');
function unary(value, calculate, derivative, name) {
  const t = asTensor(value);
  charge(t.data.length);
  const data = new Float64Array(t.data.length);
  for (let i = 0; i < data.length; i++) data[i] = finite(calculate(t.data[i]), `${name} result`);
  return result(data, t.shape, t.dtype, [t], (g, accumulate) => {
    charge(g.length * 2);
    const gt = new Float64Array(g.length);
    for (let i = 0; i < g.length; i++) gt[i] = finite(g[i] * derivative(t.data[i], data[i]), `${name} gradient`);
    accumulate(t, gt);
  });
}
export const neg = v => unary(v, x => -x, () => -1, 'negate');
export const relu = v => unary(v, x => Math.max(0, x), x => x > 0 ? 1 : 0, 'relu');
export const tanh = v => unary(v, Math.tanh, (_, y) => 1 - y * y, 'tanh');
const logistic = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export const sigmoid = v => unary(v, logistic, (_, y) => y * (1 - y), 'sigmoid');
export const exp = v => unary(v, Math.exp, (_, y) => y, 'exp');
export const log = v => unary(v, Math.log, x => 1 / x, 'log');

export function sum(value) {
  const t = asTensor(value);
  charge(t.data.length);
  let total = 0;
  for (const x of t.data) total = finite(total + x, 'Sum');
  return result([total], [], t.dtype, [t], (g, accumulate) => {
    charge(t.data.length);
    accumulate(t, new Float64Array(t.data.length).fill(g[0]));
  });
}
export function mean(value) {
  const t = asTensor(value);
  if (!t.data.length) throw new RangeError('Mean of an empty tensor is undefined');
  return div(sum(t), t.data.length);
}
export function reshape(value, shape) {
  const t = requireTensor(value);
  if (checkShape(shape) !== t.data.length) throw new RangeError('Reshape must preserve element count');
  return result(t.data, shape, t.dtype, [t], (g, accumulate) => accumulate(t, g));
}
export function transpose(value) {
  const t = requireTensor(value);
  if (t.shape.length !== 2) throw new RangeError('Transpose currently requires a matrix');
  const [rows, cols] = t.shape;
  charge(t.data.length);
  const data = new Float64Array(t.data.length);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) data[j * rows + i] = t.data[i * cols + j];
  return result(data, [cols, rows], t.dtype, [t], (g, accumulate) => {
    charge(t.data.length);
    const gt = new Float64Array(t.data.length);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) gt[i * cols + j] = g[j * rows + i];
    accumulate(t, gt);
  });
}
export function matmul(a, b) {
  [a, b] = pair(a, b);
  if (![1, 2].includes(a.shape.length) || ![1, 2].includes(b.shape.length)) throw new RangeError('Matmul supports vectors and matrices, without batch dimensions');
  const av = a.shape.length === 1, bv = b.shape.length === 1;
  const rows = av ? 1 : a.shape[0], inner = av ? a.shape[0] : a.shape[1], cols = bv ? 1 : b.shape[1];
  if (inner !== b.shape[0]) throw new RangeError('Matmul inner dimensions must match');
  const shape = av && bv ? [] : av ? [cols] : bv ? [rows] : [rows, cols];
  checkShape(shape);
  charge(rows * cols * Math.max(1, inner) * 2);
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    let total = 0;
    for (let k = 0; k < inner; k++) total += a.data[i * inner + k] * b.data[k * cols + j];
    data[i * cols + j] = finite(total, 'Matmul result');
  }
  return result(data, shape, a.dtype === 'f64' || b.dtype === 'f64' ? 'f64' : 'f32', [a, b], (g, accumulate) => {
    charge(rows * cols * Math.max(1, inner) * 4);
    const ga = a.requiresGrad ? new Float64Array(a.data.length) : null;
    const gb = b.requiresGrad ? new Float64Array(b.data.length) : null;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) for (let k = 0; k < inner; k++) {
      const gi = g[i * cols + j], ai = i * inner + k, bi = k * cols + j;
      if (ga) ga[ai] = finite(ga[ai] + gi * b.data[bi], 'Matmul gradient');
      if (gb) gb[bi] = finite(gb[bi] + gi * a.data[ai], 'Matmul gradient');
    }
    if (ga) accumulate(a, ga);
    if (gb) accumulate(b, gb);
  });
}
export function softmax(value) {
  const t = requireTensor(value);
  if (!t.shape.length || t.shape.at(-1) === 0) throw new RangeError('Softmax requires a nonempty last dimension');
  const width = t.shape.at(-1);
  charge(t.data.length * 4);
  const data = new Float64Array(t.data.length);
  for (let start = 0; start < data.length; start += width) {
    let maximum = -Infinity, denominator = 0;
    for (let j = 0; j < width; j++) maximum = Math.max(maximum, t.data[start + j]);
    for (let j = 0; j < width; j++) { data[start + j] = Math.exp(t.data[start + j] - maximum); denominator += data[start + j]; }
    for (let j = 0; j < width; j++) data[start + j] /= denominator;
  }
  return result(data, t.shape, t.dtype, [t], (g, accumulate) => {
    charge(t.data.length * 4);
    const gt = new Float64Array(t.data.length);
    for (let start = 0; start < data.length; start += width) {
      let dot = 0;
      for (let j = 0; j < width; j++) dot += g[start + j] * data[start + j];
      for (let j = 0; j < width; j++) gt[start + j] = finite(data[start + j] * (g[start + j] - dot), 'Softmax gradient');
    }
    accumulate(t, gt);
  });
}
export const mse = (a, b) => mean(pow(sub(a, b), 2));
export function crossEntropy(logits, labels) {
  const t = requireTensor(logits);
  if (t.shape.length !== 1 && t.shape.length !== 2) throw new RangeError('Cross entropy expects logits [classes] or [batch, classes]');
  const rows = t.shape.length === 1 ? 1 : t.shape[0], cols = t.shape.at(-1);
  if (!rows || !cols) throw new RangeError('Cross entropy requires a nonempty batch and class dimension');
  const l = labels instanceof Tensor ? labels : tensor(labels, 'f64');
  if (l.shape.length !== 1 || l.data.length !== rows || l.requiresGrad) throw new RangeError('Cross entropy labels must be a non-differentiable vector of batch class indices');
  for (const y of l.data) if (!Number.isSafeInteger(y) || y < 0 || y >= cols) throw new RangeError('Cross entropy label index is outside the class dimension');
  charge(t.data.length * 5);
  const probability = new Float64Array(t.data.length);
  let loss = 0;
  for (let i = 0; i < rows; i++) {
    const start = i * cols;
    let maximum = -Infinity, denominator = 0;
    for (let j = 0; j < cols; j++) maximum = Math.max(maximum, t.data[start + j]);
    for (let j = 0; j < cols; j++) { probability[start + j] = Math.exp(t.data[start + j] - maximum); denominator += probability[start + j]; }
    loss = finite(loss + (maximum - t.data[start + l.data[i]]) + Math.log(denominator), 'Cross entropy loss');
    for (let j = 0; j < cols; j++) probability[start + j] /= denominator;
  }
  return result([loss / rows], [], t.dtype, [t], (g, accumulate) => {
    charge(t.data.length * 3);
    const gt = new Float64Array(t.data.length);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) gt[i * cols + j] = finite((probability[i * cols + j] - (j === l.data[i] ? 1 : 0)) * g[0] / rows, 'Cross entropy gradient');
    accumulate(t, gt);
  });
}
export function argmax(value) {
  const t = requireTensor(value);
  if (!t.data.length) throw new RangeError('Argmax requires a nonempty tensor');
  charge(t.data.length);
  let index = 0;
  for (let i = 0; i < t.data.length; i++) {
    finite(t.data[i]);
    if (t.data[i] > t.data[index]) index = i;
  }
  return index;
}
export function sliceFirst(value, index) {
  const t = requireTensor(value);
  if (!t.shape.length || !Number.isSafeInteger(index) || index < 0 || index >= t.shape[0]) throw new RangeError('Tensor index must be an integer within the first dimension');
  const shape = t.shape.slice(1), width = checkShape(shape), start = index * width;
  return result(t.data.subarray(start, start + width), shape, t.dtype, [t], (g, accumulate) => {
    charge(t.data.length + g.length);
    const gt = new Float64Array(t.data.length);
    gt.set(g, start);
    accumulate(t, gt);
  });
}
export function detach(value) {
  const t = requireTensor(value);
  return fromFlat(t.data, t.shape, t.dtype, false);
}
export function serialize(value) {
  const t = requireTensor(value);
  charge(t.data.length);
  let offset = 0;
  function nested(depth) {
    charge(1);
    if (depth === t.shape.length) return finite(t.data[offset++]);
    return Array.from({ length: t.shape[depth] }, () => nested(depth + 1));
  }
  return { type: 'tensor', shape: [...t.shape], dtype: t.dtype, data: nested(0) };
}

export function backward(loss) {
  const root = requireTensor(loss);
  if (root.shape.length !== 0) throw new RangeError('Backward requires a scalar loss');
  if (!root.requiresGrad) throw new RangeError('Loss is not connected to a differentiable parameter');
  finite(root.data[0], 'Loss');
  const order = [], visited = new Set(), active = new Set(), stack = [[root, false]];
  while (stack.length) {
    charge(1);
    const [node, expanded] = stack.pop();
    if (expanded) { active.delete(node); visited.add(node); order.push(node); continue; }
    if (visited.has(node)) continue;
    if (active.has(node)) throw new Error('Differentiation graph contains a cycle');
    active.add(node);
    stack.push([node, true]);
    for (let i = node._parents.length - 1; i >= 0; i--) {
      const p = node._parents[i];
      if (p._version !== node._versions[i]) throw new Error('A tensor was updated after this loss was computed; recompute the loss before backward');
      if (p.requiresGrad && !visited.has(p)) stack.push([p, false]);
    }
  }
  const gradients = new Map([[root, new Float64Array([1])]]);
  function accumulate(node, values) {
    if (!node.requiresGrad) return;
    charge(values.length);
    const previous = gradients.get(node);
    if (previous) {
      for (let i = 0; i < values.length; i++) previous[i] = finite(previous[i] + values[i], 'Accumulated gradient');
    } else {
      for (const x of values) finite(x, 'Gradient');
      gradients.set(node, values);
    }
  }
  const pending = [];
  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i], g = gradients.get(node);
    if (!g) continue;
    if (node._backward) node._backward(g, accumulate);
    else {
      charge(g.length);
      const combined = new Float64Array(g.length);
      for (let j = 0; j < g.length; j++) combined[j] = finite(g[j] + (node.grad?.[j] ?? 0), 'Leaf gradient');
      pending.push([node, combined]);
    }
  }
  // Commit only after every gradient passed validation.
  for (const [node, gradient] of pending) node.grad = gradient;
  return root;
}
function parameters(params) {
  const list = params instanceof Tensor ? [params] : params;
  if (!Array.isArray(list)) throw new TypeError('Parameters must be a tensor list');
  const unique = new Set();
  for (const p of list) {
    requireTensor(p);
    if (!p.requiresGrad || p._parents.length) throw new TypeError('Training parameters must be differentiable leaf tensors');
    if (unique.has(p)) throw new TypeError('A parameter cannot occur twice');
    unique.add(p);
  }
  return list;
}
export function zeroGrad(params) {
  const list = parameters(params);
  for (const p of list) p.grad = null;
}
export function step(params, learningRate) {
  const list = parameters(params);
  finite(learningRate, 'Learning rate');
  if (learningRate <= 0) throw new RangeError('Learning rate must be positive');
  const count = list.reduce((n, p) => n + p.data.length, 0);
  charge(count * 3);
  const updates = [];
  for (const p of list) {
    if (!(p.grad instanceof Float64Array) || p.grad.length !== p.data.length) throw new Error('Every parameter requires a gradient before step');
    const data = p.dtype === 'f32' ? new Float32Array(p.data.length) : new Float64Array(p.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = finite(p.data[i], 'Parameter') - learningRate * finite(p.grad[i], 'Parameter gradient');
      finite(data[i], 'Updated parameter');
    }
    updates.push(data);
  }
  // Validation is atomic across all parameters; invalid updates change nothing.
  for (let i = 0; i < list.length; i++) {
    list[i].data.set(updates[i]);
    list[i]._version++;
    list[i]._parents = [];
    list[i]._versions = [];
    list[i]._backward = null;
  }
}
