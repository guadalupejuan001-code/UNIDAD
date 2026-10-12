"""Own-engine factorial probe; ideal prior replies are separately disclosed aids."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ia_propia.conversation.runtime import ConversationEngine


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--expected-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent
    freeze = json.loads((folder / 'freeze.json').read_text())
    for name, expected in freeze['filesSha256'].items():
        assert sha(folder / name) == expected, name
    cases = json.loads((folder / 'cases.json').read_text())
    ideal = {r['episodeId']: r['previousAssistantReplies'] for r in
             json.loads((folder / 'ideal-history.json').read_text())['episodes']}
    source = folder.parents[1] / 'src/ia_propia/conversation'
    model_dir, output = Path(args.model).resolve(), Path(args.output)
    pins = {str(p): sha(p) for p in [Path(__file__), folder / 'freeze.json',
        *[folder / name for name in freeze['filesSha256']], *sorted(source.glob('*.py')),
        *[model_dir / name for name in ('manifest.json', 'config.json', 'tokenizer.json', 'weights.safetensors')]]}
    assert sha(model_dir / 'manifest.json') == args.expected_manifest
    engine = ConversationEngine(model_dir)
    assert engine.manifest['training']['trainedFrom'] in ('random_weights', 'own_checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'Never overwrite earlier real generations'
    generation = cases['generation']
    started = time.monotonic()
    report = {'schemaVersion': 1, 'purpose': 'development_factorial_not_final_acceptance',
              'startedAtUtc': datetime.now(timezone.utc).isoformat(),
              'modelId': engine.model_id, 'modelManifest': engine.manifest,
              'sourceFilesSha256': pins, 'generation': generation, 'episodes': []}
    for episode in cases['episodes']:
        branches = ('actual', 'ideal') if len(episode['turns']) > 1 else ('actual',)
        for branch in branches:
            history = []
            observed = {key: value for key, value in episode.items() if key != 'turns'}
            observed.update({'historyCondition': branch, 'turns': []})
            if branch == 'ideal':
                assert len(ideal[episode['id']]) == len(episode['turns'])
            for index, turn in enumerate(episode['turns']):
                if branch == 'ideal':
                    history = []
                    for previous in range(index):
                        history.extend([{'role': 'user', 'content': episode['turns'][previous]['user']},
                                        {'role': 'assistant', 'content': ideal[episode['id']][previous]}])
                history.append({'role': 'user', 'content': turn['user']})
                row = {'id': f'{episode["id"]}/t{index + 1}', 'user': turn['user'],
                       'messages': [dict(message) for message in history],
                       'pastIdealRepliesFed': index if branch == 'ideal' else 0,
                       'currentOrFutureIdealReplyFed': False, 'reply': None, 'error': None}
                try:
                    reply = engine.reply(history, max_new_tokens=generation['maxNewTokens'],
                                         temperature=generation['temperature'], seed=generation['seed'])
                    row['reply'] = reply
                    row['actualTextSha256'] = hashlib.sha256(reply['text'].encode()).hexdigest()
                    if branch == 'actual':
                        history.append({'role': 'assistant', 'content': reply['text']})
                except (ValueError, OSError, RuntimeError) as error:
                    row['error'] = {'type': type(error).__name__, 'message': str(error)}
                observed['turns'].append(row)
                write(output, report | {'episodes': report['episodes'] + [observed]})
            report['episodes'].append(observed)
    assert all(sha(path) == expected for path, expected in pins.items())
    rows = [t for episode in report['episodes'] for t in episode['turns']]
    assert len(rows) == 80
    report.update({'completedAtUtc': datetime.now(timezone.utc).isoformat(),
                   'elapsedSeconds': time.monotonic() - started, 'frozenInputsUnchanged': True,
                   'summary': {'responses': len(rows), 'technicalErrors': sum(r['error'] is not None for r in rows),
                       'emptyReplies': sum(r['reply'] is not None and not r['reply']['text'].strip() for r in rows),
                       'eos': sum(r['reply'] is not None and r['reply']['stopReason'] == 'eos' for r in rows),
                       'tokenBudgetStops': sum(r['reply'] is not None and r['reply']['stopReason'] == 'token_budget' for r in rows),
                       'promptCuts': sum(r['reply'] is not None and r['reply']['discardedPromptTokens'] > 0 for r in rows)}})
    write(output, report)
    lines = []
    for episode in report['episodes']:
        lines.append(episode['id'] + ' [' + episode['historyCondition'] + ']')
        for row in episode['turns']:
            lines.extend(['Usuario: ' + row['user'], 'Unidad: ' + (row['reply']['text'] if row['reply'] else ''),
                          'Error: ' + str(row['error']) if row['error'] else '', ''])
    output.with_suffix('.txt').write_text('\n'.join(lines))
    print(json.dumps({'modelId': engine.model_id, 'summary': report['summary'], 'sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
