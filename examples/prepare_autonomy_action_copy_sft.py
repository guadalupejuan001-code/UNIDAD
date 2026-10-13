"""DATA ONLY: real file receipts for broader action-argument filename copying.

The existing v1 fixture executor and exact prompt/native encoder are reused.
No model weights, training, decoding filters, or tool/schema changes occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import unicodedata


DATASET = "unidad-autonomy-action-copy-v2"
V1_PACKAGE = Path("examples/learning-repair-20261013/autonomy-action-sft-v1")
PARENT = Path("models/learning-repair-20261013/autonomy/action2048-final-best256")
PARENT_ID = "unidad-conversation-e202eafc8f11bcda7587db496c6e7ac1670efbd612685c465658ae2f07bd54fa"
BASE = Path(__file__).with_name("prepare_autonomy_action_sft.py")
spec = importlib.util.spec_from_file_location("unidad_action_fixture_v1_reuse", BASE)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


TRAIN_WORDS = (
    "bruma", "nácar", "cauce", "helecho", "ámbar", "marea", "juncal", "cumbre",
    "greda", "brote", "zafiro", "musgo", "trigal", "alba", "gruta", "espiga")
VAL_WORDS = ("abedul", "páramo", "ébano", "hayedo", "ocaso", "líquen", "estepa", "jícara")
TRAIN_QUALIFIERS = ("azul", "claro", "alto", "seco", "leve", "gris", "sur", "norte")
VAL_QUALIFIERS = ("ocre", "opaco", "bajo", "fino", "mate", "liso", "rojo", "negro")
REPORT_TRAIN = ("r", "síntesis", "balance", "registro", "resumen", "salida", "recuento", "evidencia")
REPORT_VAL = ("dictamen", "archivo", "bitácora", "revisión", "extracto", "lectura", "valoración", "constancia")
EXTENSIONS = (".txt", ".md", ".csv")
FORBIDDEN_EXACT = ("uno.txt", "dos.txt", "modelo.json", "k.txt")
FORBIDDEN_DEMO_GOAL = "Analiza uno.txt y dos.txt y guarda un informe sin alterar originales."


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def filename(words, qualifier, number, style, extension):
    if style == 0:
        stem = words + str(number)
    elif style == 1:
        stem = words + "-" + qualifier + "-" + str(number)
    elif style == 2:
        stem = words + " " + qualifier + " " + str(number)
    else:
        stem = words + " " + str(number)
    return stem + extension


def build_task_specs():
    tasks = []
    for split, count in (("train", 64), ("validation", 8)):
        words = TRAIN_WORDS if split == "train" else VAL_WORDS
        qualifiers = TRAIN_QUALIFIERS if split == "train" else VAL_QUALIFIERS
        reports = REPORT_TRAIN if split == "train" else REPORT_VAL
        for index in range(count):
            number = index + (11 if split == "train" else 83)
            first = filename(words[index % len(words)], qualifiers[index % len(qualifiers)], number,
                             index % 4, EXTENSIONS[index % 3])
            second = filename(words[(index + 5) % len(words)], qualifiers[(index + 3) % len(qualifiers)], number + 1,
                              (index + 2) % 4, EXTENSIONS[(index + 1) % 3])
            # Separate lexical pool and independent permutation avoid file-stem/report shortcuts.
            report_number = (101 + (index * 37 + 19) % 64) if split == "train" else (301 + (index * 5 + 3) % 8)
            report = reports[(index * 3 + 1) % len(reports)] + ("-" if index % 2 else " ") + str(report_number) + ".json"
            world = ("Tervelia" if split == "train" else "Ausnoria") + str(number)
            mode = index % 4
            first_tool = "read_file" if mode == 0 else "inspect_file"
            second_tool = "read_file" if mode == 1 else "inspect_file"
            order = [first, second] if index % 2 == 0 else [second, first]
            if split == "train":
                templates = (
                    "Necesito un informe en «{r}». Lee el contenido de «{a}», inspecciona «{b}» y conserva las fuentes.",
                    "Revisa las medidas de «{a}» y lee «{b}»; deja el informe en «{r}» sin cambiar ningún archivo de entrada.",
                    "Comprueba «{a}» y después «{b}» por sus medidas. Guarda los resultados en «{r}»; preserva ambas entradas.",
                    "Para este cierre, inspecciona primero «{a}» y luego «{b}». Crea «{r}» con la evidencia y conserva sus bytes.",
                    "Quiero dejar constancia en «{r}»: lee «{a}», mide «{b}» y mantén intactos los archivos de origen.",
                    "Documenta estos materiales: inspecciona «{a}», lee «{b}» y escribe «{r}». Las entradas deben quedar intactas.",
                    "Examina las medidas de «{a}» antes que las de «{b}»; prepara «{r}» a partir de esa evidencia sin alterar las fuentes.",
                    "Haz una revisión verificable de «{a}», seguida de «{b}», y deposita el informe «{r}» sin modificar las entradas.")
            else:
                templates = (
                    "Deja una constancia «{r}» después de consultar el texto de «{a}» y tomar medidas de «{b}»; las fuentes quedan intactas.",
                    "Toma las medidas de «{a}», consulta el texto de «{b}» y entrega «{r}». Mantén intactas ambas fuentes.",
                    "Obtén primero las medidas de «{a}», a continuación las de «{b}», y registra la revisión en «{r}»; conserva el origen.",
                    "Entrega «{r}» basándote en medidas de «{a}» y luego «{b}». No introduzcas cambios en los archivos de entrada.",
                    "Para dejar evidencia «{r}», consulta «{a}» como texto y toma medidas de «{b}»; conserva sus bytes originales.",
                    "Necesito evidencia «{r}» de las medidas de «{a}» y del contenido de «{b}». Conserva las entradas como están.",
                    "Registra en «{r}» las medidas de «{a}» y, posteriormente, las de «{b}»; el material de origen debe permanecer intacto.",
                    "Comprueba las medidas de «{a}», continúa con «{b}» y deja una constancia «{r}» manteniendo intacto el material de entrada.")
            template_index = index % len(templates)
            goal = templates[template_index].format(a=order[0], b=order[1], r=report)
            group = f"autonomy-copy-v2:{split}:{world.lower()}"
            task_files = [first, second] if index % 3 else [second, first]
            contents = {}
            for position, name in enumerate(task_files):
                quantity = number * 3 + position * 7
                if name.endswith(".csv"):
                    content = f"sector,unidades\n{world},{quantity}\n"
                elif name.endswith(".md"):
                    content = f"# {world}\nMaterial ficticio: {quantity} unidades.\n"
                else:
                    content = f"Sector ficticio {world}: inventario de {quantity} unidades.\n"
                contents[name] = content
            tasks.append({"split": split, "groupId": group, "family": "copy-v2-world:" + world,
                          "world": world, "phraseTemplateId": f"copy-v2:{split}:goal:{template_index}",
                          "phraseTemplate": templates[template_index], "goal": goal,
                          "files": contents, "report": report,
                          "actions": [fixture.tool(first_tool, {"filename": order[0]}, "Leer" if first_tool == "read_file" else "Medir"),
                                      fixture.tool(second_tool, {"filename": order[1]}, "Leer" if second_tool == "read_file" else "Medir"),
                                      fixture.tool("save_report", {"filename": report, "files": task_files}, "Fuentes observadas")]})
    return tasks


def validate_tasks(tasks, previous):
    train, validation = ([t for t in tasks if t["split"] == s] for s in ("train", "validation"))
    if len(train) != 64 or len(validation) != 8:
        raise ValueError("Exactly64 TRAIN/8 validation whole tasks required")
    if len({t["groupId"] for t in tasks}) != 72:
        raise ValueError("Duplicate task identity")
    for key in ("groupId", "family", "world", "phraseTemplateId", "phraseTemplate"):
        if {t[key] for t in train} & {t[key] for t in validation}:
            raise ValueError("Split leakage: " + key)
    folds = {s: {normalized(name) for t in tasks if t["split"] == s for name in [*t["files"], t["report"]]}
             for s in ("train", "validation")}
    if folds["train"] & folds["validation"]:
        raise ValueError("Filename/report split leakage after accent/case normalization")
    old_names = {normalized(name) for r in previous for name in
                 [*json.loads(r["messages"][1]["content"])["task"]["task_inputs"]["files"],
                  json.loads(r["messages"][1]["content"])["task"]["task_inputs"]["report"]]}
    if (folds["train"] | folds["validation"]) & old_names:
        raise ValueError("Previous corpus filename overlap")
    for task in tasks:
        for name in task["files"]:
            if not 6 <= len(name) <= 24 or Path(name).suffix not in EXTENSIONS:
                raise ValueError("Filename length/extension outside registered distribution: " + name)
            if not 1 <= len(re.split(r"[ -]+", Path(name).stem)) <= 3:
                raise ValueError("Filename requires1–3 stem words: " + name)
        if not 6 <= len(task["report"]) <= 20:
            raise ValueError("Report basename length exceeded: " + task["report"])
        # Related demo basenames remain excluded even if accents, case or extensions change.
        for name in [*task["files"], task["report"]]:
            stem_words = re.split(r"[ _-]+", normalized(Path(name).stem))
            if set(stem_words) & {"uno", "dos", "modelo", "model", "k"}:
                raise ValueError("Related observed demo basename leaked")
        raw = fixture.compact(task)
        if any(needle.casefold() in raw.casefold() for needle in FORBIDDEN_EXACT) or FORBIDDEN_DEMO_GOAL in raw:
            raise ValueError("Observed demo/model error became training data")
    return {"wholeTaskFamilyWorldAndGoalTemplatesDisjoint": True,
            "normalizedTrainValidationBasenamesDisjoint": True, "previousV1BasenamesDisjoint": True,
            "exactExcludedNeedles": list(FORBIDDEN_EXACT), "excludedDemoGoal": FORBIDDEN_DEMO_GOAL,
            "relatedExcludedStemTokens": ["uno", "dos", "modelo", "model", "k"],
            "needleMatchesInTaskSpecs": 0, "testRowsRead": 0}


def copy_statistics(records, tokenizer):
    totals = Counter()
    names, value_token_ids, token_sequences, characters, bpe_lengths, name_lengths = set(), set(), set(), set(), [], []
    for record in records:
        text = record["messages"][-1]["content"]
        action = json.loads(text)
        arguments = action["arguments"]
        values = [arguments["filename"], *arguments.get("files", [])]
        user_content = record["messages"][1]["content"]
        user_ids = tokenizer.encode(user_content, add_special_tokens=False).ids
        user_id_set = set(user_ids)
        observed = json.loads(user_content)["task"]["task_inputs"]
        if action["tool"] == "save_report":
            if arguments["filename"] != observed["report"] or arguments["files"] != observed["files"]:
                raise ValueError("Report arguments do not copy this actual task")
        elif arguments["filename"] not in observed["files"]:
            raise ValueError("Read/inspection filename is outside actual task inputs")
        argument_start = text.index('"arguments":') + len('"arguments":')
        argument_raw = fixture.compact(arguments)
        if text[argument_start:argument_start + len(argument_raw)] != argument_raw:
            raise ValueError("Compact argument span mismatch")
        spans = []
        for value in values:
            serialized = json.dumps(value, ensure_ascii=False)
            cursor = argument_raw.index(serialized)
            spans.append((argument_start + cursor + 1, argument_start + cursor + len(serialized) - 1))
            encoded_value = tokenizer.encode(value, add_special_tokens=False).ids
            names.add(value)
            value_token_ids.update(encoded_value)
            token_sequences.add(tuple(encoded_value))
            if serialized not in user_content:
                raise ValueError("Argument value absent from the actual current USER payload")
            totals["argumentQuotedStringsExactlyPresentInCurrentUser"] += 1
            contiguous = any(user_ids[start:start + len(encoded_value)] == encoded_value
                             for start in range(len(user_ids) - len(encoded_value) + 1))
            totals["standaloneFilenameBpeSequencesContiguousInCurrentUser"] += contiguous
            characters.update(value)
            bpe_lengths.append(len(encoded_value))
            name_lengths.append(len(value))
            totals["argumentValueOccurrences"] += 1
        encoding = tokenizer.encode(text, add_special_tokens=False)
        payload_indices = [index for index, (start, end) in enumerate(encoding.offsets)
                           if any(start < right and end > left for left, right in spans)]
        payload_tokens = len(payload_indices)
        totals["filenameTargetBpeTokensPresentInCurrentUser"] += sum(encoding.ids[index] in user_id_set for index in payload_indices)
        totals["filenameTargetBpeTokensAbsentFromCurrentUser"] += sum(encoding.ids[index] not in user_id_set for index in payload_indices)
        target_tokens = len(encoding.ids) + 1  # Native final EOS is also a target.
        totals["assistantTargetTokensIncludingEos"] += target_tokens
        totals["filenameValueOverlappingBpeTargetTokens"] += payload_tokens
        totals["remainingActionTargetTokensIncludingEos"] += target_tokens - payload_tokens
    return {**dict(totals), "distinctBasenames": len(names), "distinctFilenameBpeTokenIds": len(value_token_ids),
            "distinctFilenameBpeSequences": len(token_sequences), "distinctFilenameCharacters": len(characters),
            "filenameBpeLengths": fixture.distribution(bpe_lengths), "basenameCharacterLengths": fixture.distribution(name_lengths),
            "filenameValueTargetFraction": totals["filenameValueOverlappingBpeTargetTokens"] / totals["assistantTargetTokensIncludingEos"],
            "allArgumentValuesCopiedFromActualTaskInputs": True,
            "method": "Whole BPE tokens whose character offsets overlap filename/report JSON string interiors; mixed boundary tokens count once. Remaining tokens include JSON keys/syntax, tool/reason values and final EOS, rather than syntax alone.",
            "modelAbilityMeasured": False}


def prepare(output, parent=PARENT):
    from tokenizers import Tokenizer
    from ia_propia.conversation.data import load_corpus, validate_record
    from ia_propia.conversation.tokenizer import validate_tokenizer
    started = time.monotonic()
    output, parent = Path(output), Path(parent)
    output.mkdir(parents=True, exist_ok=False)
    old_manifest_path = V1_PACKAGE / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    old_corpus_path = V1_PACKAGE / old_manifest["file"]["path"]
    frozen_old_files = {str(p): fixture.sha(p.read_bytes()) for p in V1_PACKAGE.iterdir() if p.is_file()}
    old_records = load_corpus(old_corpus_path)
    if any(r["split"] == "test" for r in old_records):
        raise ValueError("Previous comparison data must contain no TEST")
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest["modelId"] != PARENT_ID or parent_manifest["config"]["context_length"] != 2048:
        raise ValueError("Matched future continuation requires exact E202 own artifact")
    tokenizer_path = parent / "tokenizer.json"
    tokenizer_sha = fixture.sha(tokenizer_path.read_bytes())
    if tokenizer_sha != parent_manifest["files"]["tokenizer.json"]["sha256"] or tokenizer_sha != old_manifest["tokenizer"]["sha256"]:
        raise ValueError("Own tokenizer provenance mismatch")
    tokenizer = validate_tokenizer(Tokenizer.from_file(str(tokenizer_path)), strict=True)
    if tokenizer.get_vocab_size() != 2048:
        raise ValueError("Require unchanged own BPE2048")
    tasks = build_task_specs()
    exclusions = validate_tasks(tasks, old_records)
    config = deepcopy(parent_manifest["training"]["config"])
    if config["max_steps"] != 256 or config["max_seconds"] != 600:
        raise ValueError("Matched native limits changed")
    fixture.write_json(output / "future-training-config.json", config)
    design = {"schemaVersion": 1, "dataset": DATASET, "status": "preregistered-before-real-fixture-execution",
              "hypothesis": "V1 learned JSON structure but did not ground the read filename in current task inputs. Broadening basename BPE sequences and natural goal order should improve copying under unchanged action SFT.",
              "interventionCount": 1, "intervention": "Data distribution breadth for exact current-task filename/report argument copying",
              "reportNames": "Separate TRAIN/validation lexical pools independent of input basenames; independent numeric permutation; only task_inputs.report authorizes target report name",
              "unchanged": ["own E202 parent", "model architecture", "tokenizer", "native assistant_last_reply loss", "generic three-tool schemas", "host validators", "generation and raw JSON validation"],
              "modelErrorUsedAsPositiveTarget": False, "oldV1BytesPreserved": True,
              "splitCounts": {"train": 192, "validation": 24, "test": 0},
              "taskCounts": {"train": 64, "validation": 8, "test": 0},
              "splitUnit": "whole fictional world/task/family and separately authored goal templates; normalized filenames/report names disjoint",
              "contextLength": 2048, "objective": "assistant_last_reply", "testUsedForTraining": False,
              "parentModelId": parent_manifest["modelId"], "parentWeightsSha256": parent_manifest["files"]["weights.safetensors"]["sha256"],
              "parentManifestSha256": fixture.sha((parent / "manifest.json").read_bytes()),
              "tokenizerSha256": tokenizer_sha, "previousCorpusSha256": old_manifest["corpusSha256"],
              "taskSpecSha256": fixture.sha(fixture.compact(tasks).encode("utf-8")),
              "futureMatchedComparison": {"status": "proposed-not-run-not-queued", "sameOwnParentForBothArms": PARENT_ID,
                  "treatment": "new copy-v2 corpus", "control": "unchanged v1 corpus rehearsal",
                  "configuration": "future-training-config.json identical for both arms; reset optimizer/scheduler equally",
                  "logicalUpdatesMaximum": 256, "nativeSecondsMaximum": 600, "singleNativeTrainer": True,
                  "evaluation": "New unseen basename/goal/task families, actual raw JSON, exact current-task authorized arguments and verified tool completion; no fixture fallback",
                  "notEvaluation": "NLL or valid JSON alone does not establish task completion or Spanish conversation competence",
                  "stop": "Retain failure evidence; no extra updates beyond finite limits without a new design"},
              "preparationBudget": {"wallSeconds": 120, "cpuThreads": 1, "rssMiB": 256, "diskMiB": 128, "networkUsed": False},
              "author_AI": True, "independentHumanReview": False, "modelWeightsLoaded": False, "trainingStarted": False}
    fixture.write_json(output / "preregister.json", design)
    records, receipts = [], []
    generic_tools = json.loads(old_records[0]["messages"][1]["content"])["tools"]
    for task in tasks:
        task_records, receipt = fixture.execute_task(task, tokenizer)
        for record in task_records:
            record["source"].update(dataset=DATASET, parentFixtureExecutor="prepare_autonomy_action_sft.execute_task",
                                   phraseTemplateId=task["phraseTemplateId"], intervention="filename_argument_copy_distribution")
            validate_record(record)
            view = json.loads(record["messages"][1]["content"])
            if view["tools"] != generic_tools:
                raise ValueError("Generic three-tool metadata changed")
            if any(needle.casefold() in fixture.compact(record["messages"]).casefold() for needle in FORBIDDEN_EXACT):
                raise ValueError("Excluded demo/model error in actual training messages")
        records.extend(task_records)
        receipts.append(receipt)
        if time.monotonic() - started > 115:
            raise TimeoutError("Data preparation wall ceiling")
    corpus_path = output / "autonomy-action-copy.train-validation.jsonl"
    corpus_path.write_text("".join(fixture.compact(r) + "\n" for r in records), encoding="utf-8")
    (output / "execution-receipts.jsonl").write_text("".join(fixture.compact(r) + "\n" for r in receipts), encoding="utf-8")
    if load_corpus(corpus_path) != records:
        raise ValueError("Canonical corpus changed")
    token_audit, copying = {}, {}
    for split in ("train", "validation"):
        rows = [r for r in records if r["split"] == split]
        token_audit[split] = fixture.audit_tokens(rows, tokenizer)
        copying[split] = copy_statistics(rows, tokenizer)
    copying["previousV1Train"] = copy_statistics([r for r in old_records if r["split"] == "train"], tokenizer)
    copying["previousV1Validation"] = copy_statistics([r for r in old_records if r["split"] == "validation"], tokenizer)
    exclusions.update({"needleMatchesInActualTrainingMessages": 0, "oldV1FileHashesBefore": frozen_old_files,
                       "oldV1BytesPreserved": all(fixture.sha(Path(p).read_bytes()) == h for p, h in frozen_old_files.items()),
                       "oldV1FamilyGroupsDisjoint": not ({r["groupId"] for r in records} & {r["groupId"] for r in old_records}),
                       "genericToolsMetadataSha256": fixture.sha(fixture.compact(generic_tools).encode("utf-8")),
                       "genericThreeToolSchemasUnchanged": True})
    if not exclusions["oldV1BytesPreserved"] or not exclusions["oldV1FamilyGroupsDisjoint"]:
        raise ValueError("Old corpus changed or groups leaked")
    qa = {"schemaVersion": 1, "tokenAudit": token_audit, "filenameCopyStatistics": copying,
          "tasksActuallyExecuted": len(receipts), "taskCompletionsVerified": sum(r["terminalStatus"] == "completed" for r in receipts),
          "actualVerifiedToolResults": sum(e["kind"] == "tool_result" and e["verification"]["verified"] for r in receipts for e in r["journal"]),
          "originalsUnchanged": all(r["originalsUnchanged"] for r in receipts), "temporaryRootsDestroyed": True,
          "elapsedSeconds": time.monotonic() - started, "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
          "torchLoaded": "torch" in sys.modules, "modelWeightsLoaded": False, "trainingStarted": False,
          "testRowsRead": 0, "learnedPlanningProven": False, "conversationCompetenceProven": False,
          "review": "Project AI and actual filesystem/code measurements, not independent human evaluation"}
    if qa["torchLoaded"] or qa["peakRssMiB"] > 256:
        raise ValueError("No-weight/RAM preparation limit violated")
    fixture.write_json(output / "qa.json", qa)
    fixture.write_json(output / "needle-exclusion-and-split-audit.json", exclusions)
    content = "".join(json.dumps(r["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    corpus_sha = fixture.sha(corpus_path.read_bytes())
    recipes = {"schemaVersion": 1, "status": "data-ready-future-comparison-proposed-not-queued-not-run",
               "parentModelPath": str(parent), "parentModelId": PARENT_ID, "optimizerResetBothArms": True,
               "trainingConfigPath": "future-training-config.json", "trainingConfigSha256": fixture.sha((output / "future-training-config.json").read_bytes()),
               "limits": {"logicalUpdates": 256, "nativeSeconds": 600, "logicalBatchSize": 2, "microBatchSize": 1},
               "singleTrainerSequentialArms": True,
               "arms": {"new-copy-v2": {"corpusPath": corpus_path.name, "corpusSha256": corpus_sha},
                        "v1-rehearsal": {"corpusPath": str(old_corpus_path), "corpusSha256": old_manifest["corpusSha256"]}},
               "admission": "Root review required before any supervisor admission; this file is a proposal, not a cycle manifest or approval"}
    fixture.write_json(output / "future-matched-recipes.json", recipes)
    manifest = {"schemaVersion": 1, "dataset": DATASET, "status": "prepared-not-trained-not-queued-not-autonomy-accepted",
                "corpusSha256": corpus_sha, "contentSha256": fixture.sha(content.encode("utf-8")),
                "contentSerialization": "sorted compact UTF-8 messages JSON followed by LF, in corpus order",
                "file": {"path": corpus_path.name, "bytes": corpus_path.stat().st_size, "sha256": corpus_sha},
                "splitCounts": {"train": 192, "validation": 24, "test": 0}, "taskCounts": {"train": 64, "validation": 8, "test": 0},
                "testUsedForTraining": False, "testUsed": False, "objective": "assistant_last_reply", "contextLength": 2048,
                "sources": [{"dataset": DATASET, "kind": "authored-synthetic", "description": fixture.SOURCE,
                             "author_AI": True, "independentHumanReview": False, "license": "CC0-1.0", "licensePath": "LICENSE-DATA.md",
                             "actualFilesystemMeasurements": True, "modelGeneratedLabels": False, "privateConversationUsed": False}],
                "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_sha, "vocabSize": 2048},
                "ownParentModelId": PARENT_ID, "previousV1CorpusSha256": old_manifest["corpusSha256"],
                "helperSha256": fixture.sha(Path(__file__).read_bytes()), "reusedFixtureHelperSha256": fixture.sha(BASE.read_bytes()),
                "sourceSha256": {name: fixture.sha(Path(name).read_bytes()) for name in
                    ("src/ia_propia/brain/autonomy_model.py", "src/ia_propia/brain/autonomy.py", "src/ia_propia/brain/file_tools.py",
                     "src/ia_propia/conversation/tokenizer.py", "src/ia_propia/conversation/training.py")},
                "preregisterSha256": fixture.sha((output / "preregister.json").read_bytes()),
                "qaSha256": fixture.sha((output / "qa.json").read_bytes()),
                "executionReceiptsSha256": fixture.sha((output / "execution-receipts.jsonl").read_bytes()),
                "splitAndExclusionAuditSha256": fixture.sha((output / "needle-exclusion-and-split-audit.json").read_bytes()),
                "futureRecipesSha256": fixture.sha((output / "future-matched-recipes.json").read_bytes()),
                "modelWeightsLoaded": False, "trainingStarted": False, "queued": False, "learnedPlanningProven": False}
    fixture.write_json(output / "manifest.json", manifest)
    (output / "LICENSE-DATA.md").write_text("# Dataset license\n\nNew project-authored fictional task text, curated fixture action labels, and real temporary-file measurements are dedicated under CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/ .\n\nAI authoring/review; no independent human certification. No private conversations, third-party text, or model-generated targets. Existing source code and tokenizer retain their original repository license.\n", encoding="utf-8")
    (output / "README.md").write_text("# Copia de argumentos de archivos v2\n\n192 TRAIN y 24 VALIDATION de 64/8 tareas completas. Las etiquetas son JSON correcto de una fixture declarada; SafeFileTools y el controlador ejecutaron 216 acciones reales, verificaron informes y conservaron originales por hash. Se reutilizó el ejecutor v1 con sus tres herramientas y esquemas genéricos sin cambios.\n\nLa intervención cambia sólo la distribución de datos para copiar nombres de archivo y de informe del objetivo y task_inputs actuales: nombres de una a tres palabras, tildes, espacios, guiones y extensiones txt/md/csv; orden del objetivo y de la lista de entradas variable. Los nombres, familias y plantillas de objetivo de VALIDATION quedan separados de TRAIN; todos son ficción, sin nombres del demo observado. Los datos v1 permanecen intactos. No se enseñó como correcto el nombre rechazado que produjo el modelo.\n\n[Diseño previo](preregister.json), [manifiesto](manifest.json), [QA y estadísticas BPE](qa.json), [exclusiones y splits](needle-exclusion-and-split-audit.json), [recibos reales](execution-receipts.jsonl), [recetas futuras](future-matched-recipes.json), [licencia](LICENSE-DATA.md). La auditoría nativa supervisa únicamente el último JSON y su EOS; cada prompt completo más target cabe en 2048. TEST no se leyó.\n\nEl contraste propuesto continúa ambos brazos desde el mismo E202 propio: v2 frente a ensayo v1, configuración idéntica, máximo 256 actualizaciones o 600 segundos nativos y un solo entrenador. Esta preparación no entrenó, no encoló y no cambió arquitectura, pérdida ni generación. Validez JSON y NLL bajas no demuestran argumentos autorizados, finalización real, generalización ni competencia conversacional en español.\n\nReproducción desde la raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python3 examples/prepare_autonomy_action_copy_sft.py --output-dir /tmp/unidad-action-copy-new`. Escribe una carpeta nueva; nunca sustituye v1.\n", encoding="utf-8")
    disk = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if disk > 128 * 1024 ** 2:
        raise ValueError("Data-only disk ceiling exceeded")
    return {"output": str(output), "corpusSha256": corpus_sha, "contentSha256": manifest["contentSha256"],
            "maxInputs": {s: token_audit[s]["maxSequenceInputs"] for s in token_audit},
            "copying": {s: {k: copying[s][k] for k in ("distinctBasenames", "distinctFilenameBpeTokenIds", "filenameValueTargetFraction")} for s in copying},
            "elapsedSeconds": qa["elapsedSeconds"], "peakRssMiB": qa["peakRssMiB"], "diskBytes": disk,
            "trainingStarted": False, "queued": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 ** 2, 128 * 1024 ** 2))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Copy-data wall ceiling")))
    signal.alarm(120)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    print(fixture.compact(prepare(args.output_dir, args.parent)))


if __name__ == "__main__":
    main()
