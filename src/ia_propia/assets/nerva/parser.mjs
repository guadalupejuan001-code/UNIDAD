const MAX_SOURCE = 200_000;
const MAX_TOKENS = 100_000;
const MAX_NODES = 100_000;
const MAX_DEPTH = 64;
const MAX_ARITY = 32;
const KEYWORDS = new Set([
  'let', 'const', 'param', 'fn', 'if', 'else', 'for', 'in', 'while',
  'return', 'break', 'continue', 'true', 'false', 'null', 'not', 'and',
  'or', 'nerva',
]);

export class LangError extends Error {
  constructor(message, { line = 1, column = 1 } = {}) {
    super(`${message} (line ${line}, column ${column})`);
    this.name = 'LangError';
    this.line = line;
    this.column = column;
  }
}

// Tokens are data only. Source is never passed to JavaScript evaluation.
export function lex(source) {
  if (typeof source !== 'string') throw new LangError('Source must be a string');
  if (source.length > MAX_SOURCE) throw new LangError(`Source exceeds ${MAX_SOURCE} characters`);
  const tokens = [];
  let position = 0;
  let line = 1;
  let column = 1;
  let grouping = 0;
  const advance = () => {
    const char = source[position++];
    if (char === '\r') {
      if (source[position] === '\n') position++;
      line++;
      column = 1;
    } else if (char === '\n') {
      line++;
      column = 1;
    } else column++;
    return char;
  };
  const emit = (type, value, atLine, atColumn) => {
    if (tokens.length >= MAX_TOKENS) throw new LangError(`Token limit of ${MAX_TOKENS} exceeded`, { line: atLine, column: atColumn });
    tokens.push({ type, value, line: atLine, column: atColumn });
  };
  while (position < source.length) {
    const char = source[position];
    if (char === ' ' || char === '\t' || char === '\f') {
      advance();
      continue;
    }
    const atLine = line;
    const atColumn = column;
    if (char === '\r' || char === '\n') {
      advance();
      if (grouping === 0) emit('newline', '\n', atLine, atColumn);
      continue;
    }
    if (char === '#') {
      while (position < source.length && source[position] !== '\n' && source[position] !== '\r') advance();
      continue;
    }
    if (/[A-Za-z_]/.test(char)) {
      const start = position;
      while (position < source.length && /[A-Za-z0-9_]/.test(source[position])) advance();
      const value = source.slice(start, position);
      emit(KEYWORDS.has(value) ? 'keyword' : 'id', value, atLine, atColumn);
      continue;
    }
    if (/[0-9]/.test(char) || (char === '.' && /[0-9]/.test(source[position + 1] ?? ''))) {
      const match = /^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/.exec(source.slice(position));
      const spelling = match[0];
      for (let index = 0; index < spelling.length; index++) advance();
      if (source[position] === 'e' || source[position] === 'E') throw new LangError('Invalid numeric exponent', { line: atLine, column: atColumn });
      const value = Number(spelling);
      if (!Number.isFinite(value)) throw new LangError('Numeric literal must be finite', { line: atLine, column: atColumn });
      emit('number', value, atLine, atColumn);
      continue;
    }
    if (char === '"' || char === "'") {
      const quote = advance();
      let value = '';
      let closed = false;
      while (position < source.length) {
        const next = source[position];
        if (next === quote) {
          advance();
          closed = true;
          break;
        }
        if (next === '\n' || next === '\r') throw new LangError('String literal cannot contain an unescaped newline', { line, column });
        if (next !== '\\') {
          value += advance();
          continue;
        }
        const escapeLine = line;
        const escapeColumn = column;
        advance();
        if (position >= source.length) break;
        const escape = advance();
        const simple = { n: '\n', r: '\r', t: '\t', b: '\b', f: '\f', '\\': '\\', '"': '"', "'": "'", '/': '/' };
        if (Object.hasOwn(simple, escape)) value += simple[escape];
        else if (escape === 'u') {
          const digits = source.slice(position, position + 4);
          if (!/^[0-9a-fA-F]{4}$/.test(digits)) throw new LangError('Expected four hexadecimal digits after \\u', { line: escapeLine, column: escapeColumn });
          value += String.fromCharCode(Number.parseInt(digits, 16));
          for (let index = 0; index < 4; index++) advance();
        } else throw new LangError(`Unknown string escape \\${escape}`, { line: escapeLine, column: escapeColumn });
      }
      if (!closed) throw new LangError('Unterminated string literal', { line: atLine, column: atColumn });
      emit('string', value, atLine, atColumn);
      continue;
    }
    const pair = source.slice(position, position + 2);
    if (['==', '!=', '<=', '>='].includes(pair)) {
      advance();
      advance();
      emit('op', pair, atLine, atColumn);
      continue;
    }
    if ('+-*/@^<>=()[]{},:;'.includes(char)) {
      advance();
      if (char === '(' || char === '[') grouping++;
      else if (char === ')' || char === ']') grouping = Math.max(0, grouping - 1);
      emit('()[]{},:;'.includes(char) ? 'punct' : 'op', char, atLine, atColumn);
      continue;
    }
    throw new LangError(`Unexpected character ${JSON.stringify(char)}`, { line: atLine, column: atColumn });
  }
  emit('eof', null, line, column);
  return tokens;
}

const PRECEDENCE = new Map([
  ['or', 20], ['and', 30],
  ['==', 40], ['!=', 40], ['<', 40], ['<=', 40], ['>', 40], ['>=', 40],
  ['+', 50], ['-', 50], ['*', 60], ['/', 60], ['@', 60], ['^', 80],
]);

class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.position = 0;
    this.depth = 0;
    this.nodes = 0;
    this.nodeDepths = new WeakMap();
  }
  peek(offset = 0) { return this.tokens[Math.min(this.position + offset, this.tokens.length - 1)]; }
  take() { return this.tokens[this.position++]; }
  is(value) {
    const token = this.peek();
    return token.type !== 'string' && token.type !== 'number' && token.value === value;
  }
  fail(message, token = this.peek()) { throw new LangError(message, token); }
  expect(value) {
    if (!this.is(value)) this.fail(`Expected ${JSON.stringify(value)}`);
    return this.take();
  }
  identifier() {
    if (this.peek().type !== 'id') this.fail('Expected an identifier');
    return this.take();
  }
  separators() {
    while (this.peek().type === 'newline' || this.is(';')) this.take();
  }
  newlines() { while (this.peek().type === 'newline') this.take(); }
  bounded(work) {
    if (++this.depth > MAX_DEPTH) this.fail(`Parse depth exceeds ${MAX_DEPTH}`);
    try { return work(); } finally { this.depth--; }
  }
  node(type, fields, token, children = []) {
    if (++this.nodes > MAX_NODES) this.fail(`AST node limit of ${MAX_NODES} exceeded`, token);
    let depth = 1;
    for (const child of children) if (child) depth = Math.max(depth, (this.nodeDepths.get(child) ?? 0) + 1);
    if (depth > MAX_DEPTH) this.fail(`AST depth exceeds ${MAX_DEPTH}`, token);
    const node = { type, ...fields, line: token.line, column: token.column };
    this.nodeDepths.set(node, depth);
    return node;
  }
  program() {
    this.separators();
    if (this.is('nerva')) {
      this.take();
      const version = this.peek();
      if (version.type !== 'number' || version.value !== 0.2) this.fail('Only the nerva 0.2 header is supported');
      this.take();
      if (this.peek().type !== 'newline' && !this.is(';') && this.peek().type !== 'eof') this.fail('Expected a separator after the version header');
      this.separators();
    }
    const body = this.statements(false);
    return { type: 'program', body };
  }
  statements(inBlock) {
    const body = [];
    this.separators();
    while (this.peek().type !== 'eof' && !(inBlock && this.is('}'))) {
      body.push(this.statement());
      if (this.peek().type === 'eof' || (inBlock && this.is('}'))) break;
      if (this.peek().type !== 'newline' && !this.is(';')) this.fail('Expected a newline or semicolon between statements');
      this.separators();
    }
    return body;
  }
  block() {
    this.newlines();
    this.expect('{');
    const body = this.statements(true);
    this.expect('}');
    return body;
  }
  statement() {
    return this.bounded(() => {
      const token = this.peek();
      if (token.type === 'keyword' && ['let', 'const', 'param'].includes(token.value)) return this.declaration();
      if (this.is('fn')) return this.functionDeclaration();
      if (this.is('if')) return this.ifStatement();
      if (this.is('for')) {
        this.take();
        const name = this.identifier().value;
        this.expect('in');
        const iterable = this.expression();
        const body = this.block();
        return this.node('for', { name, iterable, body }, token, [iterable, ...body]);
      }
      if (this.is('while')) {
        this.take();
        const test = this.expression();
        const body = this.block();
        return this.node('while', { test, body }, token, [test, ...body]);
      }
      if (this.is('return')) {
        this.take();
        const expr = this.peek().type === 'newline' || this.peek().type === 'eof' || this.is(';') || this.is('}') ? null : this.expression();
        return this.node('return', { expr }, token, [expr]);
      }
      if (this.is('break') || this.is('continue')) {
        this.take();
        return this.node(token.value, {}, token);
      }
      if (token.type === 'id' && this.peek(1).type === 'op' && this.peek(1).value === '=') {
        this.take();
        this.take();
        const expr = this.expression();
        return this.node('assign', { name: token.value, expr }, token, [expr]);
      }
      const expr = this.expression();
      return this.node('expr', { expr }, token, [expr]);
    });
  }
  declaration() {
    const token = this.take();
    const name = this.identifier().value;
    let shape = null;
    if (this.is(':')) {
      this.take();
      this.expect('Tensor');
      this.expect('[');
      shape = [];
      if (!this.is(']')) {
        while (true) {
          const dimension = this.peek();
          if (dimension.type !== 'number' || !Number.isSafeInteger(dimension.value) || dimension.value < 0) this.fail('Tensor dimensions must be nonnegative integers');
          if (dimension.value > 1_000_000) this.fail('Tensor dimension exceeds the element limit');
          shape.push(this.take().value);
          if (shape.length > 8) this.fail('Tensor rank exceeds 8', dimension);
          if (!this.is(',')) break;
          this.take();
          if (this.is(']')) break;
        }
      }
      this.expect(']');
    }
    this.expect('=');
    const expr = this.expression();
    return this.node('let', { kind: token.value, name, shape, expr }, token, [expr]);
  }
  functionDeclaration() {
    const token = this.take();
    const name = this.identifier().value;
    this.expect('(');
    const params = [];
    if (!this.is(')')) {
      while (true) {
        const parameter = this.identifier();
        if (params.includes(parameter.value)) this.fail(`Duplicate parameter ${parameter.value}`, parameter);
        params.push(parameter.value);
        if (params.length > MAX_ARITY) this.fail(`Functions accept at most ${MAX_ARITY} parameters`, parameter);
        if (!this.is(',')) break;
        this.take();
        if (this.is(')')) break;
      }
    }
    this.expect(')');
    const body = this.block();
    return this.node('fn', { name, params, body }, token, body);
  }
  ifStatement() {
    const token = this.expect('if');
    const test = this.expression();
    const then = this.block();
    let otherwise = [];
    // A newline can join an else to its if, but belongs to the outer
    // statement list when there is no else.
    const saved = this.position;
    this.newlines();
    if (this.is('else')) {
      this.take();
      this.newlines();
      otherwise = this.is('if') ? [this.bounded(() => this.ifStatement())] : this.block();
    } else this.position = saved;
    return this.node('if', { test, then, else: otherwise }, token, [test, ...then, ...otherwise]);
  }
  expression(minimum = 0) {
    return this.bounded(() => {
      let left = this.prefix();
      while (true) {
        const token = this.peek();
        if (token.type === 'punct' && token.value === '(' && 100 >= minimum) {
          this.take();
          const args = [];
          if (!this.is(')')) {
            while (true) {
              args.push(this.expression());
              if (args.length > MAX_ARITY) this.fail(`Calls accept at most ${MAX_ARITY} arguments`, token);
              if (!this.is(',')) break;
              this.take();
              if (this.is(')')) break;
            }
          }
          this.expect(')');
          left = this.node('call', { callee: left, args }, token, [left, ...args]);
          continue;
        }
        if (token.type === 'punct' && token.value === '[' && 100 >= minimum) {
          this.take();
          const index = this.expression();
          this.expect(']');
          left = this.node('index', { target: left, index }, token, [left, index]);
          continue;
        }
        const precedence = token.type === 'op' || token.type === 'keyword' ? PRECEDENCE.get(token.value) : undefined;
        if (precedence === undefined || precedence < minimum) break;
        this.take();
        const right = this.expression(token.value === '^' ? precedence : precedence + 1);
        left = this.node('binary', { op: token.value, left, right }, token, [left, right]);
      }
      return left;
    });
  }
  prefix() {
    const token = this.take();
    if (token.type === 'number' || token.type === 'string') return this.node('literal', { value: token.value }, token);
    if (['true', 'false', 'null'].includes(token.value)) return this.node('literal', { value: token.value === 'null' ? null : token.value === 'true' }, token);
    if (token.type === 'id') return this.node('id', { name: token.value }, token);
    if (token.value === '+' || token.value === '-' || token.value === 'not') {
      const arg = this.expression(70);
      return this.node('unary', { op: token.value, arg }, token, [arg]);
    }
    if (token.value === '(') {
      const expr = this.expression();
      this.expect(')');
      return expr;
    }
    if (token.value === '[') {
      const items = [];
      if (!this.is(']')) {
        while (true) {
          items.push(this.expression());
          if (!this.is(',')) break;
          this.take();
          if (this.is(']')) break;
        }
      }
      this.expect(']');
      return this.node('list', { items }, token, items);
    }
    this.fail('Expected an expression', token);
  }
}

export function parse(source) {
  return new Parser(lex(source)).program();
}
