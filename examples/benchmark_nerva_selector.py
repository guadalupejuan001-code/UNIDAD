"""Finite read-only profiling of the real Python/Node boundary (not the model)."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    import torch
    from ia_propia.conversation.nerva_decoder import NervaDecoder
    torch.set_num_threads(1)
    controls = {'top_k': 0, 'top_p': 1, 'repetition_penalty': 1.1}
    rows = []
    for size in (2048, 8192):
        logits = torch.sin(torch.arange(size, dtype=torch.float32) * .31)
        values = {'scores': logits.tolist(), 'excluded': [0], 'generated': [1, 2, 2],
                  'temperature': 0, 'draw': 0, **controls}
        def encode():
            return (json.dumps(values, allow_nan=False, separators=(',', ':')) + '\n').encode('ascii')
        payload = encode()
        for repeat in range(5):
            modes = (False, True) if repeat % 2 == 0 else (True, False)
            for fast in modes:
                start = time.perf_counter()
                decoder = NervaDecoder(42, optimized=fast)
                with decoder:
                    enter = time.perf_counter() - start
                    start = time.perf_counter()
                    expected = decoder.select(logits, [0], [1, 2, 2], temperature=0, **controls)
                    first = time.perf_counter() - start
                    for _ in range(10):
                        assert decoder.select(logits, [0], [1, 2, 2], temperature=0, **controls) == expected
                    functions = {
                        'tolist': lambda: logits.detach().cpu().tolist(),
                        'json_encode': encode,
                        'ipc_preencoded': lambda: decoder.executor.submit(decoder._exchange, payload).result(timeout=decoder.timeout),
                        'complete_select': lambda: decoder.select(logits, [0], [1, 2, 2], temperature=0, **controls),
                    }
                    stages = {}
                    names = list(functions)
                    for stage in names[repeat % 4:] + names[:repeat % 4]:
                        fn = functions[stage]
                        start = time.perf_counter()
                        for _ in range(50):
                            result = fn()
                            if stage in ('ipc_preencoded', 'complete_select'):
                                assert result == expected
                        stages[stage] = (time.perf_counter() - start) * 1000 / 50
                    start = time.perf_counter()
                close = time.perf_counter() - start
                rows.append({'vocabularySize': size, 'repeat': repeat, 'optimized': fast, 'token': expected,
                             'payloadBytes': len(payload), 'enterSeconds': enter,
                             'firstSelectionSeconds': first, 'closeSeconds': close,
                             'millisecondsPerDecision': stages})
    summary = []
    for size in (2048, 8192):
        for fast in (False, True):
            group = [r for r in rows if r['vocabularySize'] == size and r['optimized'] == fast]
            summary.append({'vocabularySize': size, 'optimized': fast,
                'medianMillisecondsPerDecision': {k: statistics.median(r['millisecondsPerDecision'][k] for r in group)
                                                  for k in group[0]['millisecondsPerDecision']},
                'medianEnterPlusFirstSelectionSeconds': statistics.median(r['enterSeconds'] + r['firstSelectionSeconds'] for r in group)})
    report = {'scope': 'selector_with_real_Python_Node_IPC_no_model_forward', 'rounds': 5, 'callsPerStage': 50,
        'note': 'Independent stage measurements; do not add their medians as an exact attribution. IPC includes JSON parsing and selection in Node.',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'node': subprocess.check_output(['node', '--version'], text=True).strip()},
        'allTokensMatch': len({(r['vocabularySize'], r['token']) for r in rows}) == 2,
        'summary': summary, 'samples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
