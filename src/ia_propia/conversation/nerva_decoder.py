"""Explicit NERVA selection for logits from the own-weight Transformer.

The neural forward pass remains Torch. A fixed NERVA program handles masking,
greedy/sampling, top-k/top-p and repetition penalties in a per-reply worker.
Nothing trains here. Node.js 18+ is required; errors never switch backends.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from importlib import resources
import json
from pathlib import Path
import random
import subprocess
import tempfile

from .tokenizer import ConversationError, number


class NervaDecoder:
    def __init__(self, seed, *, node="node", timeout=10, optimized=False):
        if type(optimized) is not bool:
            raise ConversationError("NERVA optimized must be boolean")
        self.optimized = optimized
        self.rng = random.Random(seed)  # local 63-bit seed, unaffected by global RNGs
        self.timeout = number(timeout, "NERVA timeout", 0.001, 60)
        self.node = node
        self.directory = None
        self.process = None
        self.executor = None

    def __enter__(self):
        if not isinstance(self.node, str) or not self.node or "\0" in self.node:
            raise ConversationError("NERVA requires a host-configured Node executable")
        try:
            self.directory = tempfile.TemporaryDirectory(prefix="unidad-nerva-decoder-")
            root = Path(self.directory.name)
            packaged = resources.files("ia_propia").joinpath("assets", "nerva")
            filenames = ["parser.mjs", "tensor.mjs", "runtime.mjs", "conversation_decoder.nv", "conversation_worker.mjs"]
            if self.optimized:
                filenames += ["token_selector.mjs", "conversation_worker_fast.mjs"]
            for filename in filenames:
                (root / filename).write_bytes(packaged.joinpath(filename).read_bytes())
            worker = "conversation_worker_fast.mjs" if self.optimized else "conversation_worker.mjs"
            self.process = subprocess.Popen([self.node, str(root / worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nerva-decoder")
            return self
        except OSError as error:
            self.__exit__(None, None, None)
            raise ConversationError(f"NERVA requires a working Node.js 18+ executable: {error}") from error

    def __exit__(self, *_):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            # Killing first unblocks a failed pipe read/write in the I/O thread.
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.process.stdin.close()
            self.process.stdout.close()
        if self.directory is not None:
            self.directory.cleanup()

    def _exchange(self, payload):
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        line = self.process.stdout.readline(4097)
        if len(line) > 4096 or not line.endswith(b"\n"):
            raise ConversationError("NERVA worker returned incomplete or oversized output")
        result = json.loads(line)
        if not isinstance(result, dict) or set(result) != {"token"} or type(result["token"]) is not int:
            detail = result.get("error", "invalid token record") if isinstance(result, dict) else "invalid output"
            raise ConversationError(f"NERVA selection failed: {detail}")
        return result["token"]

    def select(self, logits, forbidden, generated, *, temperature, **controls):
        values = {"scores": logits.detach().cpu().tolist(), "excluded": forbidden,
                  "generated": generated, "temperature": temperature,
                  "draw": self.rng.random() if temperature else 0, **controls}
        payload = (json.dumps(values, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
        if len(payload) > 8 * 1024 * 1024:
            raise ConversationError("NERVA decoder input exceeds its byte budget")
        try:
            token = self.executor.submit(self._exchange, payload).result(timeout=self.timeout)
        except TimeoutError as error:
            raise ConversationError(f"NERVA selection exceeded {self.timeout:g} seconds") from error
        except (OSError, ValueError) as error:
            raise ConversationError(f"NERVA selection failed: {error}") from error
        if not 0 <= token < len(logits) or token in forbidden:
            raise ConversationError("NERVA selected a forbidden or out-of-vocabulary token")
        return token
