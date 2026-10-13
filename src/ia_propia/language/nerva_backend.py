"""Optional own-weight character-RNN inference through canonical NERVA 0.2.

Python validates the existing artifact and encodes its prompt. The fixed .nv
program performs every recurrent tanh step, output projection and greedy token
decision with float64 tensors. This is not the conversational C7 backend, a
training path, a speed optimization, or a substitution with an external model.
Node.js 18+ is an explicit host dependency; errors never fall back to NumPy.
"""
from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import subprocess
import tempfile

from .model import LanguageError, encode, integer, number, validate_artifact


NERVA_VERSION = "0.2.0"
RUNTIME_FILES = ("parser.mjs", "tensor.mjs", "runtime.mjs", "nerva.mjs")
PROGRAM = "language_rnn.nv"


def _execute(inputs, *, node, timeout):
    """Run packaged code in an isolated temporary directory, without a shell."""
    if not isinstance(node, str) or not node or "\0" in node:
        raise LanguageError("NERVA node executable must be a nonempty path")
    timeout = number(timeout, "NERVA timeout", positive=True)
    packaged = resources.files("ia_propia").joinpath("assets", "nerva")
    with tempfile.TemporaryDirectory(prefix="unidad-nerva-") as directory:
        root = Path(directory)
        # Copy resources so installed wheels and importers agree on module paths.
        # Only the fixed interpreter and program are copied; model inputs are
        # ephemeral and the entire directory is removed on success or failure.
        try:
            for filename in (*RUNTIME_FILES, PROGRAM):
                (root / filename).write_bytes(packaged.joinpath(filename).read_bytes())
            data = root / "inputs.json"
            data.write_text(json.dumps(inputs, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError) as error:
            raise LanguageError(f"Cannot prepare packaged NERVA backend: {error}") from error
        command = [node, str(root / "nerva.mjs"), "run", str(root / PROGRAM),
                   "--json", "--inputs", str(data)]
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     encoding="utf-8", timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise LanguageError(f"NERVA inference exceeded its {timeout:g}-second timeout") from error
        except OSError as error:
            raise LanguageError(f"NERVA backend requires a working Node.js 18+ executable: {error}") from error
        if process.returncode:
            message = process.stderr.strip()
            try:
                decoded = json.loads(message)
                message = decoded.get("error", message) if isinstance(decoded, dict) else message
            except ValueError:
                pass
            raise LanguageError(f"NERVA inference failed: {message or process.returncode}")
        try:
            result = json.loads(process.stdout)
            if (not isinstance(result, dict) or result.get("version") != NERVA_VERSION
                    or not isinstance(result.get("steps"), int) or isinstance(result["steps"], bool)
                    or not 0 < result["steps"] <= 200_000):
                raise ValueError("unexpected runtime version or step count")
            records = result.get("records")
            if not isinstance(records, list) or len(records) != 1:
                raise ValueError("expected exactly one continuation record")
            record = records[0]
            if (not isinstance(record, dict) or record.get("kind") != "print"
                    or record.get("version") != "0.2"
                    or not isinstance(record.get("values"), list) or len(record["values"]) != 2
                    or record["values"][0] != "continuation"
                    or not isinstance(record["values"][1], str)):
                raise ValueError("invalid continuation record")
            return record["values"][1]
        except (ValueError, KeyError, TypeError) as error:
            raise LanguageError(f"Invalid output from NERVA backend: {error}") from error


def generate_nerva(artifact, prompt="", *, length=200, temperature=0.0,
                   seed=42, node="node", timeout=30):
    """Generate from an existing UNIDAD character artifact in greedy mode.

    Return the same model identity, provenance and experimental metadata as the
    NumPy generator, plus executionBackend. Float64 numerical parity is tested,
    not mathematically guaranteed for every artifact or near-tied argmax. The
    canonical interpreter's operation and step limits remain active; reaching a
    limit raises LanguageError rather than silently switching implementations.
    The recorded seed is retained; this greedy program uses no random operation.
    """
    validate_artifact(artifact)
    integer(length, "length", 0, 4096)
    integer(seed, "seed")
    temperature = number(temperature, "temperature")
    if not 0 <= temperature <= 2:
        raise LanguageError("temperature must be in [0, 2]; zero means greedy")
    if temperature != 0:
        raise LanguageError("NERVA backend currently requires temperature=0 (greedy); sampling is not implemented")
    if not isinstance(prompt, str) or len(prompt) > 4096:
        raise LanguageError("Prompt must be a string of at most 4,096 characters")
    vocabulary = artifact["tokenizer"]
    initial = encode(prompt, vocabulary).tolist()
    effective_prompt = prompt
    if not initial:
        probabilities = artifact["training"]["unigram"]["probabilities"]
        # Preserve the original train-unigram fallback and lowest-index tie.
        token = max(range(1, len(probabilities)), key=probabilities.__getitem__)
        initial = [token]
        effective_prompt = vocabulary["tokens"][token]
    weights = artifact["weights"]
    continuation = _execute({
        "Wxh": weights["Wxh"], "Whh": weights["Whh"],
        "Why": weights["Why"][1:], "bh": weights["bh"], "by": weights["by"][1:],
        "unknown_projection": weights["Why"][0], "unknown_bias": weights["by"][0],
        "tokens": initial, "character_tokens": vocabulary["tokens"], "count": length,
    }, node=node, timeout=timeout)
    if (len(continuation) != length
            or any(character not in vocabulary["tokens"][1:] for character in continuation)):
        raise LanguageError("NERVA continuation has an invalid length or emitted character")
    return {"modelId": artifact["modelId"], "artifactSha256": artifact["artifactSha256"],
            "prompt": prompt, "effectivePrompt": effective_prompt, "continuation": continuation,
            "generatedCharacters": length, "seed": seed, "temperature": temperature,
            "experimental": True, "instructionFollowingValidated": False,
            "workerCapabilityValidated": False, "executionBackend": "nerva-0.2"}
