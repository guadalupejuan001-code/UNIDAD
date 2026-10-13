"""Project configuration joins inference and learning without learning from requests."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading


MAX_PROJECT_BYTES = 262_144
GIB = 1024 ** 3
PROFILES = {
    "15m": {"vocab_size": 2048, "d_model": 384, "n_layers": 8, "n_heads": 8,
            "context_length": 2048},
    "small": {"vocab_size": 8192, "d_model": 256, "n_layers": 6, "n_heads": 8,
              "context_length": 512},
    "50m": {"vocab_size": 8192, "d_model": 512, "n_layers": 14, "n_heads": 8,
            "context_length": 1024, "parameter_budget": 50_000_000},
    "100m": {"vocab_size": 8192, "d_model": 640, "n_layers": 19, "n_heads": 10,
             "context_length": 1024, "parameter_budget": 100_000_000},
    "1b": {"vocab_size": 16384, "d_model": 1536, "n_layers": 34, "n_heads": 24,
           "context_length": 2048, "parameter_budget": 1_000_000_000},
}
STAGE_OBJECTIVES = {"foundation": "causal_lm", "conversation": "assistant_last_reply",
                    "adaptation": "assistant_last_reply"}


def _strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate GNOSIS configuration key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite GNOSIS JSON")))


def _keys(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ValueError("Unknown or missing GNOSIS configuration fields")


def _text(value, label):
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"{label} must be bounded explicit text")
    return value


def _positive(value, label, maximum):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite positive number") from error
    if not math.isfinite(number) or not 0 < number <= maximum:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _overlap(first, second):
    return first == second or first in second.parents or second in first.parents


def _has_symlink(path):
    return path.is_symlink() or any(parent.is_symlink() for parent in path.parents)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path, *, limit=4_194_304):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("GNOSIS requires bounded regular artifact metadata")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("GNOSIS artifact metadata exceeds the byte limit")
    result = _strict_json(raw)
    if type(result) is not dict:
        raise ValueError("GNOSIS artifact metadata must be an object")
    return result


def _memory_available():
    """Best-effort Linux available memory, respecting container memory limits."""
    limits = []
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        limits.append(int(fields["MemAvailable"].split()[0]) * 1024)
    except (OSError, ValueError, KeyError):
        pass
    for limit_file, used_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                  ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit, used = int(Path(limit_file).read_text()), int(Path(used_file).read_text())
            limits.append(max(0, limit - used))
        except (OSError, ValueError):
            pass
    return min(limits) if limits else None


class GnosisProject:
    """One lazy Brain; stages delegate to the existing resumable trainer."""

    def __init__(self, config, *, config_dir):
        _keys(config, ("schemaVersion", "projectRoot", "brain", "learning"))
        if type(config["schemaVersion"]) is not int or config["schemaVersion"] != 1:
            raise ValueError("Unsupported GNOSIS schema")
        self._root = (Path(config_dir) / _text(config["projectRoot"], "projectRoot")).resolve()
        # Older host configs retain Torch decoding. NERVA is an explicit host
        # option; the measured faster Torch path remains the supplied default.
        config = deepcopy(config)
        if type(config["brain"]) is dict:
            config["brain"].setdefault("conversation_decoder", "torch")
            config["brain"].setdefault("conversation_cache", True)
        _keys(config["brain"], ("state_dir", "language_backend", "conversation_model", "expected_model_id",
                                "conversation_decoder", "conversation_cache"),
              ("conversation_ensemble", "language_model", "attention_model", "conversation_selector"))
        brain = config["brain"]
        for key in ("language_model", "attention_model", "conversation_selector"):
            if brain.get(key) is not None:
                _text(brain[key], key)
        ensemble = brain.get("conversation_ensemble")
        if brain.get("conversation_selector") is not None and ensemble is None:
            raise ValueError("A coordination selector requires a host-configured ensemble")
        self._catalog = None
        if ensemble is not None:
            _keys(ensemble, ("catalog", "primary"), ("max_candidates", "min_candidates", "max_resident",
                "max_seconds", "total_new_tokens", "max_score_calls", "critic_count",
                "max_scoring_tokens", "max_sliding_score_tokens"))
            _text(ensemble["catalog"], "ensemble catalog")
            _text(ensemble["primary"], "ensemble primary")
            from ..conversation.catalog import load_catalog
            from ..conversation.ensemble import EnsembleBudget
            EnsembleBudget(**{key: value for key, value in ensemble.items() if key not in ("catalog", "primary")})
            self._catalog = load_catalog(self.path(ensemble["catalog"]), self._root)
            matching = [entry for entry in self._catalog if entry["name"] == ensemble["primary"]]
            if len(matching) != 1 or len(self._catalog) < 2:
                raise ValueError("GNOSIS ensemble requires a catalogued primary and at least two models")
            if (brain["conversation_model"] is None or
                    self.path(matching[0]["path"]) != self.path(brain["conversation_model"]) or
                    matching[0]["modelId"] != brain["expected_model_id"]):
                raise ValueError("GNOSIS ensemble primary must match the selected artifact and identity")
        _text(brain["state_dir"], "state_dir")
        if type(brain["language_backend"]) is not str or brain["language_backend"] not in ("numpy", "nerva"):
            raise ValueError("The host must select the language backend")
        if (type(brain["conversation_decoder"]) is not str or brain["conversation_decoder"] not in ("torch", "nerva", "nerva-fast")
                or type(brain["conversation_cache"]) is not bool):
            raise ValueError("The host must select the conversation decoder and boolean cache")
        if brain["conversation_model"] is None:
            if brain["expected_model_id"] is not None:
                raise ValueError("A model identity requires a selected conversation artifact")
        else:
            _text(brain["conversation_model"], "conversation_model")
            _text(brain["expected_model_id"], "expected_model_id")
        learning = config["learning"]
        _keys(learning, ("profile", "memory_budget_gib", "storage_budget_gib", "stages"))
        if type(learning["profile"]) is not str or learning["profile"] not in PROFILES:
            raise ValueError("Unknown GNOSIS model profile")
        for key in ("memory_budget_gib", "storage_budget_gib"):
            _positive(learning[key], key, 10000)
        _keys(learning["stages"], tuple(STAGE_OBJECTIVES))
        from ..conversation.training import _configs
        for name, stage in learning["stages"].items():
            _keys(stage, ("corpus", "corpus_sha256", "run_dir", "initial_model", "training_config"))
            _text(stage["run_dir"], "run_dir")
            if stage["corpus"] is not None:
                _text(stage["corpus"], "corpus")
            digest = stage["corpus_sha256"]
            if (stage["corpus"] is None) != (digest is None):
                raise ValueError("Corpus path and SHA-256 must be configured together")
            if digest is not None and (type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Corpus SHA-256 must identify the exact training data")
            if stage["initial_model"] is not None:
                _text(stage["initial_model"], "initial_model")
            if name == "foundation" and stage["initial_model"] is not None:
                raise ValueError("Foundation starts from random own weights")
            if name != "foundation" and stage["initial_model"] is None:
                raise ValueError("Later stages require an explicit own checkpoint")
            _, training = _configs(None, stage["training_config"])
            if training.get("objective", "assistant_last_reply") != STAGE_OBJECTIVES[name]:
                raise ValueError("Stage objective differs from the declared lifecycle")
            if training.get("partition_policy") != "train_validation_only":
                raise ValueError("GNOSIS training must leave TEST outside its loader")
        paths = [self.path(s["run_dir"]) for s in learning["stages"].values()]
        if any(_overlap(first, second) for i, first in enumerate(paths) for second in paths[i + 1:]):
            raise ValueError("Learning stages need distinct, non-overlapping run directories")
        if any(_overlap(self.path(brain["state_dir"]), directory) for directory in paths):
            raise ValueError("Brain state must remain outside learning run directories")
        reference = self.path(brain["conversation_model"]) if brain["conversation_model"] else None
        if reference is not None and any(_overlap(reference, p) for p in paths):
            raise ValueError("Learning runs cannot overwrite the selected reference artifact")
        if any(_overlap(self.path(brain[key]), run) for key in ("language_model", "attention_model", "conversation_selector")
               if brain.get(key) is not None for run in paths):
            raise ValueError("Learning runs cannot overwrite selected auxiliary artifacts")
        if self._catalog is not None and any(_overlap(self.path(entry["path"]), run) for entry in self._catalog for run in paths):
            raise ValueError("Learning runs cannot overwrite any catalogued conversation artifact")
        self._config = deepcopy(config)
        self._brain = None
        self._brain_lock = threading.RLock()

    @property
    def config(self):
        return deepcopy(self._config)

    def path(self, value):
        return (self._root / value).resolve()

    @property
    def brain(self):
        with self._brain_lock:
            return self._load_brain()

    def _load_brain(self):
        if self._brain is None:
            from ..brain import Brain
            config = self._config["brain"]
            selected = self.path(config["conversation_model"]) if config["conversation_model"] else None
            # Reject an unexpected manifest before allocating model tensors.
            if selected is not None:
                metadata = _metadata(selected / "manifest.json")
                if metadata.get("modelId") != config["expected_model_id"]:
                    raise ValueError("Selected conversation model differs from the GNOSIS identity")
            ensemble_options = {}
            ensemble = config.get("conversation_ensemble")
            if ensemble is not None:
                ensemble_options = {"conversation_models": [{"name": entry["name"],
                    "model_dir": self.path(entry["path"]), "expected_model_id": entry["modelId"],
                    "decoder_backend": config["conversation_decoder"], "use_cache": config["conversation_cache"]}
                    for entry in self._catalog], "ensemble_primary": ensemble["primary"],
                    "ensemble_budget": {key: value for key, value in ensemble.items() if key not in ("catalog", "primary")}}
            brain = Brain(self.path(config["state_dir"]), conversation_model=selected,
                          conversation_selector=self.path(config["conversation_selector"]) if config.get("conversation_selector") else None,
                          language_model=self.path(config["language_model"]) if config.get("language_model") else None,
                          attention_model=self.path(config["attention_model"]) if config.get("attention_model") else None,
                          language_backend=config["language_backend"], conversation_decoder=config["conversation_decoder"],
                          conversation_cache=config["conversation_cache"], **ensemble_options)
            if selected is not None and brain.models.capabilities()["models"]["conversation"]["modelId"] != config["expected_model_id"]:
                raise ValueError("Loaded model identity differs from the GNOSIS identity")
            self._brain = brain
        return self._brain

    def connect(self, body_id, *, tools=(), memory_scope=None):
        return self.brain.connect(body_id, tools=tools, memory_scope=memory_scope)

    def _stage(self, name):
        if name not in STAGE_OBJECTIVES:
            raise ValueError("Unknown GNOSIS learning stage")
        return deepcopy(self._config["learning"]["stages"][name])

    def _learning_model(self, nominal, selected, *, resume):
        """Inspect verified metadata before admitting any inherited model tensors."""
        from ..conversation.configuration import count_config_parameters, validate_config
        from ..conversation.training import _checkpoint_files
        blockers = []
        model = nominal
        try:
            if resume:
                directory = self.path(selected["run_dir"])
                status = _metadata(directory / "status.json")
                checkpoint = _checkpoint_files(directory, status.get("lastCheckpoint"))
                state = _metadata(checkpoint / "state.json", limit=67_108_864)
                model = validate_config(state.get("model_config"))
            elif selected["initial_model"] is not None:
                metadata = _metadata(self.path(selected["initial_model"]) / "manifest.json")
                model = validate_config(metadata.get("config"))
            else:
                return model, blockers
        except (ValueError, OSError, KeyError, TypeError):
            return nominal, ["resume_checkpoint_missing_or_invalid" if resume else "previous_stage_artifact_missing_or_invalid"]
        structural = ("d_model", "n_layers", "n_heads", "context_length", "ffn_multiplier", "dropout")
        if (any(model[key] != nominal[key] for key in structural)
                or model["vocab_size"] > nominal["vocab_size"]
                or count_config_parameters(model) > nominal.get("parameter_budget", 20_000_000)):
            blockers.append("inherited_architecture_differs_from_profile")
        return model, blockers

    def plan(self, stage="foundation", *, resume=False):
        if type(resume) is not bool:
            raise ValueError("resume must be boolean")
        from ..conversation.configuration import count_config_parameters, validate_config
        from ..conversation.training import _configs
        learning, selected = self._config["learning"], self._stage(stage)
        nominal = validate_config({**PROFILES[learning["profile"]], "dropout": 0.0, "seed": 20261013})
        model, blockers = self._learning_model(nominal, selected, resume=resume)
        _, training = _configs(None, selected["training_config"])
        count = count_config_parameters(model)
        micro = training.get("micro_batch_size", training["batch_size"])
        context, dimensions = model["context_length"], model["d_model"]
        # Deliberately conservative admission estimates, not measured peaks.
        persistent = count * 16  # FP32 parameters, gradients, two AdamW moments.
        activations = 4 * micro * model["n_layers"] * (context * dimensions * (12 + 2 * model["ffn_multiplier"])
                                                        + model["n_heads"] * context * context)
        projection = 12 * micro * context * model["vocab_size"]
        # Keep the unresolved path while checking links; resolving first hides them.
        corpus = self._root / selected["corpus"] if selected["corpus"] else None
        corpus_size = corpus.stat().st_size if corpus is not None and corpus.is_file() else 0
        # Keep the conservative admission allowance for either loader: mmap
        # removes retained text/windows, but tokenizer fitting, metadata and the
        # shuffled order still grow with data. This is not a measured peak.
        data_memory = 80 * corpus_size
        data_loading = training.get("data_loading", "memory")
        estimate = persistent + activations + projection + data_memory + GIB
        cache_storage = 32 * corpus_size if data_loading == "mmap" else 0
        storage = count * 4 * 12 + corpus_size + cache_storage  # checkpoints, Adam states, export, cache.
        available_memory = _memory_available()
        parent = self.path(selected["run_dir"])
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        free_storage = shutil.disk_usage(parent).free
        if corpus is None or not corpus.is_file():
            blockers.append("corpus_not_configured_or_missing")
        elif _has_symlink(corpus) or _digest(corpus) != selected["corpus_sha256"]:
            blockers.append("corpus_identity_mismatch")
        if estimate > learning["memory_budget_gib"] * GIB:
            blockers.append("memory_budget_exceeded")
        if available_memory is not None and estimate > available_memory * 0.8:
            blockers.append("available_memory_insufficient")
        if storage > learning["storage_budget_gib"] * GIB:
            blockers.append("storage_budget_exceeded")
        if storage > free_storage * 0.8:
            blockers.append("available_storage_insufficient")
        initial = self.path(selected["initial_model"]) if selected["initial_model"] else None
        return {"schemaVersion": 1, "stage": stage, "profile": learning["profile"], "modelConfig": model,
                "parameterCount": count, "trainingConfig": training,
                "initialization": "resume_checkpoint" if resume else ("own_checkpoint" if initial else "random"),
                "memory": {"adamwPersistentBytes": persistent, "activationEstimateBytes": activations,
                           "projectionEstimateBytes": projection, "corpusEstimateBytes": data_memory,
                           "admissionEstimateBytes": estimate, "budgetBytes": int(learning["memory_budget_gib"] * GIB),
                           "availableBytes": available_memory, "measuredPeak": False},
                "storage": {"estimateBytes": storage, "freeBytes": free_storage,
                            "tokenizedCacheEstimateBytes": cache_storage},
                "dataLoading": data_loading,
                "execution": "local_cpu", "blockers": blockers, "canStart": not blockers,
                "weightsCreated": False, "conversationQualityValidated": False, "automaticLearning": False,
                "priority": ["broader_spanish_foundation", "new_dialogues_and_retention", "larger_capacity"],
                "limitations": ["resource_estimates_are_not_peak_guarantees",
                                "mmap_metadata_order_and_tokenizer_still_use_memory" if data_loading == "mmap"
                                else "corpus_and_windows_are_still_in_memory",
                                "gpu_training_not_implemented", "nerva_conversation_selection_is_host_configured", "larger_profiles_are_not_trained_models"]}

    def train(self, stage, *, budget_seconds, resume=False):
        _positive(budget_seconds, "invocation budget", 86400)
        if type(resume) is not bool:
            raise ValueError("resume must be boolean")
        plan = self.plan(stage, resume=resume)
        if not plan["canStart"]:
            raise ValueError("GNOSIS admission rejected: " + ", ".join(plan["blockers"]))
        selected = self._stage(stage)
        from ..conversation.training import train
        initial = self.path(selected["initial_model"]) if selected["initial_model"] else None
        return train(self.path(selected["corpus"]), self.path(selected["run_dir"]),
                     model_config=None if initial or resume else plan["modelConfig"], training_config=selected["training_config"],
                     budget_seconds=budget_seconds, resume=resume,
                     initial_model=initial if not resume else None,
                     expected_corpus_sha256=selected["corpus_sha256"],
                     expected_model_config=plan["modelConfig"] if initial or resume else None)

    def status(self, stage):
        from ..conversation.jobs import status
        return status(self.path(self._stage(stage)["run_dir"]))

    def export(self, stage):
        from ..conversation.training import export_best
        directory = self.path(self._stage(stage)["run_dir"])
        return export_best(directory, directory / "artifact")


def load_project(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_PROJECT_BYTES:
        raise ValueError("GNOSIS requires a bounded regular local configuration file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_PROJECT_BYTES + 1)
    if len(raw) > MAX_PROJECT_BYTES:
        raise ValueError("GNOSIS configuration exceeds the byte limit")
    return GnosisProject(_strict_json(raw), config_dir=path.parent)
