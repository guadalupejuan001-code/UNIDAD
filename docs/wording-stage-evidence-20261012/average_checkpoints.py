"""One prospectively fixed average; no new training and no fresh-test feedback."""
from copy import deepcopy
import json,time
from pathlib import Path
import torch
from ia_propia.conversation.artifacts import load_artifact,save_artifact
root=Path(__file__).resolve().parent
labels=['varied-s2-last','varied-s3-last']
plan={'createdAtUnix':time.time(),'parents':labels,'coefficients':[0.5,0.5],
      'hypothesis':'Adjacent same-run checkpoints have different generation failures under a constant learning rate. One arithmetic weight average may reduce that instability.',
      'candidatesAllowed':1,'extraTraining':False,'freshTransferOpened':False,
      'developmentSelection':'Compare the same known 40 turns and paraphrase diagnostics. Keep prior best unless this candidate improves F without critical count above C7.',
      'inspiration':'https://arxiv.org/abs/1803.05407',
      'limitation':'Inspired by weight averaging; not a reproduction of the cited SGD/SWA experiments, and no guarantee of benefit for this small AdamW transformer.'}
(root/'averaging-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
assert not (root/'candidate-selection.json').exists()
torch.set_num_threads(1)
model,tokenizer,first=load_artifact(root/labels[0])
other,other_tokenizer,second=load_artifact(root/labels[1])
assert first['config']==second['config']
assert first['files']['tokenizer.json']==second['files']['tokenizer.json']
assert first['training']['lineage']==second['training']['lineage']
assert first['training']['corpusSha256']==second['training']['corpusSha256']
left=model.state_dict();right=other.state_dict();assert left.keys()==right.keys()
averaged={name:(left[name]+right[name])*0.5 for name in left}
model.load_state_dict(averaged,strict=True)
training=deepcopy(second['training'])
training['selectionSplit']='development_weight_average'
training['stepMeaning']='Latest source checkpoint step; this is a derived average, not one trained checkpoint.'
training['derivation']={'method':'arithmetic_weight_average','coefficients':[0.5,0.5],
 'parentModelIds':[first['modelId'],second['modelId']],
 'parentSteps':[first['training']['selectedStep'],second['training']['selectedStep']],
 'parentWeightsSha256':[first['files']['weights.safetensors']['sha256'],second['files']['weights.safetensors']['sha256']],
 'optimizerForDerivedWeights':False}
saved=save_artifact(root/'varied-s3-mean',model,tokenizer,training=training)
_,_,loaded=load_artifact(root/'varied-s3-mean');assert loaded['modelId']==saved['modelId']
(root/'averaging-result.json').write_text(json.dumps({'modelId':saved['modelId'],'derivation':training['derivation']},indent=2)+'\n')
status=json.loads((root/'varied-run/status.json').read_text())
(root/'stage-closed.json').write_text(json.dumps({'closedAtUnix':time.time(),'lastStep':status['step'],
 'elapsedNativeSeconds':status['elapsedSeconds'],'trainingClosed':True,'unusedBudgetSeconds':max(0,1200-status['elapsedSeconds']),
 'reason':'Third segment showed development regression; no fourth training segment. A single cheap weight-average comparison is evaluated before freezing the candidate.'},indent=2)+'\n')
print(saved['modelId'],flush=True)
