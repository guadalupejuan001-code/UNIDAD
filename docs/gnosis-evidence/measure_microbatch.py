"""Small fabricated CPU saved-storage probe; no corpus or optimizer update."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / 'src'))
    import torch
    from ia_propia.conversation.model import ConversationModel, assistant_token_loss
    from ia_propia.conversation import training
    torch.set_num_threads(1)
    config = {'vocab_size': 2048, 'd_model': 64, 'n_layers': 4, 'n_heads': 4,
              'context_length': 128, 'dropout': 0.0, 'ffn_multiplier': 4, 'seed': 17}
    inputs = torch.arange(7, 7 + 8 * 128).reshape(8, 128)
    labels = (inputs + 1) % 2048
    labels[:, :64] = -100

    class Saved:
        def __init__(self, tensor, counter):
            self.tensor, self.counter = tensor, counter
            self.key = tensor.untyped_storage().data_ptr()
            refs = counter['storageRefs']
            if self.key not in refs:
                size = tensor.untyped_storage().nbytes()
                refs[self.key] = [0, size]
                counter['live'] += size
                counter['peak'] = max(counter['peak'], counter['live'])
            refs[self.key][0] += 1

        def __del__(self):
            refs = self.counter['storageRefs']
            refs[self.key][0] -= 1
            if not refs[self.key][0]:
                self.counter['live'] -= refs[self.key][1]
                del refs[self.key]

    def measure(micro):
        model = ConversationModel(config)
        stores = {p.untyped_storage().data_ptr() for p in model.parameters()}
        counter = {'live': 0, 'peak': 0, 'storageRefs': {}}

        def pack(tensor):
            return tensor if tensor.untyped_storage().data_ptr() in stores else Saved(tensor, counter)

        def unpack(value):
            return value.tensor if isinstance(value, Saved) else value

        started = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            if micro:
                loss = training._microbatch_backward(model, inputs, labels,
                    {'micro_batch_size': 1, 'projection_mode': 'supervised'})
            else:
                loss = assistant_token_loss(model.selected_logits(inputs, labels != -100), labels,
                                            projection_mode='supervised')
                loss.backward()
        result = {'mode': 'micro_batch_size=1' if micro else 'legacy_logical_batch_size=8',
                  'nonparameterSavedStoragePeakBytes': counter['peak'],
                  'nonparameterSavedStorageLiveAfterBackwardBytes': counter['live'],
                  'logicalLoss': float(loss.detach()),
                  'secondsSingleUncontrolledTrial': time.perf_counter() - started,
                  'parameters': model.parameter_count}
        return result, model

    full, a = measure(False)
    micro, b = measure(True)
    difference = max(float((x.grad - y.grad).abs().max()) for x, y in zip(a.parameters(), b.parameters()))
    assert full['nonparameterSavedStorageLiveAfterBackwardBytes'] == micro['nonparameterSavedStorageLiveAfterBackwardBytes'] == 0
    assert abs(full['logicalLoss'] - micro['logicalLoss']) < 1e-5
    assert difference < 1e-5
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {'schemaVersion': 1,
              'purpose': 'bounded-fabricated-CPU-activation-retention-measurement-not-language-training',
              'measurement': 'peak unique nonparameter storage bytes held by autograd saved-tensor references; excludes parameters/optimizer/untracked transient tensors; not process RSS',
              'torchVersion': str(torch.__version__), 'threads': 1, 'modelConfig': config,
              'logicalBatchSize': 8, 'microBatchSize': 1, 'sequenceLength': 128,
              'targetsPerWindow': 64, 'dropout': 0.0, 'full': full, 'micro': micro,
              'savedStoragePeakReductionFraction': 1 - micro['nonparameterSavedStoragePeakBytes'] / full['nonparameterSavedStoragePeakBytes'],
              'maxAbsoluteGradientDifference': difference, 'speedAdvantageClaimed': False,
              'newCorpusTrainingRun': False, 'oldBudgetsRenewed': False,
              'trainingSourceSha256': sha(training.__file__), 'runnerSha256': sha(__file__)}
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
