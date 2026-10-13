"""Bounded native own-weight stage with explicitly frozen inputs, preserving full states and endpoint."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ia_propia.conversation import training
from ia_propia.conversation.artifacts import save_artifact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', required=True)
    parser.add_argument('--arm', required=True, )
    args = parser.parse_args()
    design_path = Path(args.design).resolve()
    design = json.loads(design_path.read_text(encoding='utf-8'))
    root = design_path.parent
    arm = design['arms'][args.arm]
    stage = root / args.arm
    stage.mkdir(parents=True, exist_ok=True)
    expected_inputs = design['frozenFilesSha256']
    for path, expected in expected_inputs.items():
        assert digest(path) == expected, path
    start = {'startedAtUtc': datetime.now(timezone.utc).isoformat(), 'arm': args.arm,
             'designSha256': digest(design_path), 'status': 'running',
             'authorization': design['authorization']}
    write(stage / 'launch.json', start)
    status = training.train(arm['corpus'], stage / 'run', initial_model=design['initialModel'],
                            model_config=design['modelConfig'],
                            training_config=design['trainingConfig'])
    assert status['status'] in ('completed', 'stopped')
    for path, expected in expected_inputs.items():
        assert digest(path) == expected, path
    cp = status['lastCheckpoint']
    model, tokenizer, state, _ = training._load_checkpoint(stage / 'run', cp)
    metadata = {'corpusSha256': state['corpus_sha256'], 'selectedStep': state['step'],
                **training._origin_metadata(state), **training._loss_metadata(state['training_config']),
                'selectionSplit': 'prospective_fixed_endpoint', 'selectedCheckpoint': cp,
                'testEvaluated': False, 'config': state['training_config'],
                'experimentDesignSha256': digest(design_path)}
    manifest = save_artifact(stage / 'endpoint', model, tokenizer, training=metadata,
                             evaluation={'purpose': 'development_not_final_acceptance'})
    training.export_best(stage / 'run', stage / 'validation-selected')
    write(stage / 'completion.json', {**start, 'status': 'completed',
        'completedAtUtc': datetime.now(timezone.utc).isoformat(), 'nativeStatus': status,
        'endpointModelId': manifest['modelId'], 'endpointManifestSha256': digest(stage / 'endpoint/manifest.json'),
        'frozenInputsUnchanged': True, 'activeTrainingAfterReturn': False})
    print(json.dumps({'arm': args.arm, 'step': state['step'], 'nativeStatus': status['status'],
                      'reason': status['reason'], 'modelId': manifest['modelId']}), flush=True)


if __name__ == '__main__':
    main()
