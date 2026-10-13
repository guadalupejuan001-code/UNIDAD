"""One bounded paired C7 experiment; no promotion or fresh acceptance claim."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/workspace/unidad-work')
OUT = Path('/workspace/unidad-results/copy-head-retention-20261013')
MODEL = ROOT / 'models/conversation-reference-c7'
SOURCE = ROOT / 'examples/conversation-focus-curriculum.jsonl'
DEV = Path('/workspace/unidad-results/current-development-cases.json')
CORPUS = OUT / 'focused-train-validation.jsonl'
DESIGN = OUT / 'design.json'
CONFIG = {'batch_size': 4, 'learning_rate': 0.0001, 'weight_decay': 0.01,
          'grad_clip': 1.0, 'max_epochs': 100, 'max_steps': 128, 'max_seconds': 600,
          'validation_every': 64, 'checkpoint_every': 64, 'patience': 100,
          'min_delta': 0.0001, 'seed': 20261013, 'warmup_steps': 16,
          'torch_threads': 1, 'window_policy': 'sliding_context',
          'loss_reduction': 'token_mean', 'projection_mode': 'supervised',
          'objective': 'assistant_last_reply', 'partition_policy': 'train_validation_only',
          'copy_supervision': {'alignment_weight': 0.2, 'gate_weight': 0.2}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def source_hashes():
    return {name: sha(ROOT / 'src/ia_propia/conversation' / name)
            for name in ('training.py', 'model.py', 'tokenizer.py', 'artifacts.py')}


def prepare():
    from tokenizers import Tokenizer
    from ia_propia.conversation.data import load_corpus
    from ia_propia.conversation.tokenizer import dialogue_tokens
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if CORPUS.exists() or DESIGN.exists():
        raise FileExistsError('Frozen experiment inputs already exist')
    source_rows = load_corpus(SOURCE)
    rows = [row for row in source_rows if row['split'] in ('train', 'validation')]
    keep_ids = {row['id'] for row in rows}
    protected = {turn['user'] for ep in json.loads(DEV.read_text())['episodes'] for turn in ep['turns']}
    collisions = [row['id'] for row in rows if any(message['role'] == 'user' and message['content'] in protected
                  for message in row['messages'])]
    if collisions:
        raise ValueError('Exposed development message overlaps selected corpus')
    raw = b''.join(line for line in SOURCE.read_bytes().splitlines(keepends=True)
                   if json.loads(line)['id'] in keep_ids)
    with CORPUS.open('xb') as stream:
        stream.write(raw)
    CORPUS.chmod(0o600)
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    inputs = [dialogue_tokens(row['messages'], tokenizer)[0] for row in rows]
    replies = [len(tokenizer.encode(row['messages'][-1]['content']).ids) + 1 for row in rows]
    if max(len(ids) - 1 for ids in inputs) > 512 or max(replies) > 128:
        raise ValueError('Selected copy-supervision windows exceed native bounds')
    write_new(DESIGN, {'schemaVersion': 1, 'createdAtUtc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'development-frozen-backbone-retention-hypothesis-not-commercial-acceptance',
        'hypothesis': 'Updating only learned copy tensors may preserve C7 backbone knowledge better than full fine tuning.',
        'model': str(MODEL), 'modelFiles': {name: sha(MODEL / name)
            for name in ('weights.safetensors', 'config.json', 'tokenizer.json', 'manifest.json')},
        'parentModelId': json.loads((MODEL / 'manifest.json').read_text())['modelId'],
        'sourceCorpus': str(SOURCE), 'sourceCorpusSha256': sha(SOURCE),
        'corpus': str(CORPUS), 'corpusSha256': sha(CORPUS), 'recordSplits': dict(Counter(row['split'] for row in rows)),
        'familySplits': dict(Counter(split for split, group in {(row['split'], row['groupId']) for row in rows})),
        'sourceKind': 'existing-project-authored-synthetic', 'originalKeptLinesByteExact': True,
        'originalIdsGroupsPartitionsTargetsPreserved': True, 'testRecordsExcluded': len(source_rows) - len(rows),
        'maximumInputTokens': max(len(ids) - 1 for ids in inputs), 'maximumReplyTargetsIncludingEos': max(replies),
        'developmentCasesSha256': sha(DEV), 'exactDevelopmentUserMessageCollisions': [],
        'armsInOrder': ['all', 'copy_head'], 'commonTrainingConfig': CONFIG,
        'singleDifference': 'trainable_parameters', 'sameInitialWeightsTokenizerAndBatchSeed': True,
        'endpointStep': 128, 'endpointSelectedBeforeGeneration': True,
        'generation': {'maxNewTokens': 160, 'temperature': 0, 'seed': 42, 'torchThreads': 1, 'actualHistory': True},
        'selection': 'Compare fixed endpoints to original C7 on exposed development; reject increased critical failures or lost prior correct replies. No promotion without fresh reserved evaluation.',
        'priorClosedBudgetsRenewed': False, 'paidServicesUsed': False, 'externalModelWeightsUsed': False,
        'personalMessagesUsed': False, 'testEvaluated': False, 'sourceSha256AtPreparation': source_hashes(),
        'runnerSha256': sha(__file__)})
    print(json.dumps({'event': 'prepared', 'designSha256': sha(DESIGN), 'corpusSha256': sha(CORPUS)}, sort_keys=True), flush=True)


def run():
    import torch
    from ia_propia.conversation import training
    from ia_propia.conversation.artifacts import save_artifact, load_artifact
    design = json.loads(DESIGN.read_text())
    if sha(CORPUS) != design['corpusSha256'] or sha(DEV) != design['developmentCasesSha256']:
        raise ValueError('Frozen experiment inputs changed')
    if any(sha(MODEL / name) != digest for name, digest in design['modelFiles'].items()):
        raise ValueError('Parent artifact changed')
    if sha(__file__) != design['runnerSha256']:
        raise ValueError('Experiment runner changed after freeze')
    torch.set_num_threads(1)
    parent, _, _ = load_artifact(MODEL)
    original = {name: value.clone() for name, value in parent.state_dict().items()}
    del parent
    started = time.monotonic()
    sources_start = source_hashes()
    receipts = []
    print(json.dumps({'event': 'started', 'pid': os.getpid(), 'processBirth': training._birth(os.getpid()),
                      'designSha256': sha(DESIGN), 'maxStepsPerArm': 128, 'maxSecondsPerArm': 600}), flush=True)
    for policy in design['armsInOrder']:
        config = {**design['commonTrainingConfig'], 'trainable_parameters': policy}
        run_dir = OUT / policy / 'run'
        start = time.monotonic()
        result = training.train(CORPUS, run_dir, initial_model=MODEL,
                                model_config=json.loads((MODEL / 'config.json').read_text()), training_config=config)
        model, tokenizer, state, payload = training._load_checkpoint(run_dir, result['lastCheckpoint'])
        metadata = {'corpusSha256': state['corpus_sha256'], 'selectedStep': state['step'],
                    **training._origin_metadata(state), **training._loss_metadata(config),
                    **training._parameter_metadata(state), 'config': state['training_config'],
                    'selectionSplit': 'prospective_fixed_endpoint', 'selectedCheckpoint': result['lastCheckpoint'],
                    'selectionReason': '128-step endpoint fixed before this paired experiment and free generation',
                    'testEvaluated': False, 'productionAccepted': False, 'designSha256': sha(DESIGN)}
        artifact = OUT / policy / 'endpoint'
        manifest = save_artifact(artifact, model, tokenizer, training=metadata)
        changed = sorted(name for name, value in model.state_dict().items()
                         if not torch.equal(value.view(torch.uint8), original[name].view(torch.uint8)))
        if policy == 'copy_head' and not set(changed) <= training._COPY_NEW_KEYS:
            raise ValueError('Frozen backbone changed')
        receipt = {'policy': policy, 'result': result, 'wallSeconds': time.monotonic() - start,
                   'modelId': manifest['modelId'], 'endpoint': str(artifact), 'endpointStep': state['step'],
                   'endpoint128Reached': state['step'] == 128, 'changedTensorNames': changed,
                   'changedParameters': sum(model.state_dict()[name].numel() for name in changed),
                   'optimizerMomentIndices': sorted(payload['optimizer']['state']),
                   'sourceSha256AtStart': sources_start, 'sourceSha256AfterArm': source_hashes(),
                   'artifactFiles': {name: sha(artifact / name) for name in design['modelFiles']},
                   'frozenBackbonePreserved': policy == 'copy_head' and not set(changed) - training._COPY_NEW_KEYS,
                   'ownLineagePreserved': manifest['training']['lineage']['parentModelId'] == design['parentModelId'],
                   'rawResponsesFabricated': False, 'commercialReplacementValidated': False}
        write_new(OUT / policy / 'receipt.json', receipt)
        receipts.append(receipt)
        print(json.dumps({'event': 'arm_completed', 'policy': policy, 'step': state['step'],
                          'seconds': receipt['wallSeconds'], 'modelId': manifest['modelId'], 'endpoint': str(artifact)}, sort_keys=True), flush=True)
        del model, tokenizer, payload
    write_new(OUT / 'training-receipt.json', {'schemaVersion': 1, 'designSha256': sha(DESIGN),
          'wallSeconds': time.monotonic() - started, 'arms': receipts,
          'sameInProcessTrainingModulesBothArms': True, 'parentArtifactUnchanged': all(sha(MODEL / name) == digest for name, digest in design['modelFiles'].items()),
          'oldBudgetsRenewed': False, 'testEvaluated': False, 'trainingStopped': True,
          'conversationCompetenceClaimed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    action = parser.parse_args().action
    prepare() if action == 'prepare' else run()
