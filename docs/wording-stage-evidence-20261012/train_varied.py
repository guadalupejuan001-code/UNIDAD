"""One new bounded stage from own weights; no closed run is restarted."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from ia_propia.conversation.training import train,export_best,_load_checkpoint,_origin_metadata,_loss_metadata
from ia_propia.conversation.artifacts import save_artifact

p=argparse.ArgumentParser()
p.add_argument('--resume',action='store_true')
p.add_argument('--segment',type=int,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
source=root.parent/'unidad-fix/pointer-v1-best'
run=root/'varied-run'
torch.set_num_threads(4)
config={'batch_size':8,'learning_rate':0.0001,'weight_decay':0.01,'grad_clip':1.0,
        'max_epochs':1000,'max_steps':1024,'max_seconds':1200,'validation_every':128,
        'checkpoint_every':128,'patience':5,'seed':83,'torch_threads':4,'warmup_steps':32,
        'window_policy':'sliding_context','objective':'assistant_last_reply','loss_reduction':'token_mean',
        'projection_mode':'supervised','partition_policy':'train_validation_only','train_precision':'cpu_bfloat16',
        'copy_supervision':{'alignment_weight':0.2,'gate_weight':0.2}}
request={'corpus_path':str(root/'varied-corpus.jsonl'),'run_dir':str(run),
         'model_config':json.loads((source/'config.json').read_text()),'training_config':config,
         'resume':a.resume,'budget_seconds':300}
if not a.resume:request['initial_model']=str(source)
plan={'createdAtUnix':time.time(),'stage':'new-wording-generalization-stage',
      'request':request,'maxAggregateNativeTrainingSeconds':1200,'maxSteps':1024,
      'segmentNativeSeconds':300,'segmentsAreSameRunWithOptimizerResume':True,
      'hypothesis':'Varied fact wording, question wording and order teach the existing model to map equivalent prompts to consistent replies, while TRAIN-only rehearsal preserves practiced forms.',
      'startingModelId':json.loads((source/'manifest.json').read_text())['modelId'],
      'reference':'C7: C7/P5/F28, critical 6, complete dialogs 0/10 on known development.',
      'target':'F <= 14/40 with critical <= 6; also report C/P and full dialogs; independent acceptance remains 36/40,8/10,zero critical.',
      'selection':'Select by real generations on declared development, not NLL alone. Fresh transfer is used only after candidate selection.',
      'sourceCommit':'a1a283caabf89b3191c8ce64c2a950e9edbb0edf plus final-position projection; beam experiment is not used for training or this evaluation.',
      'corpusSha256':hashlib.sha256((root/'varied-corpus.jsonl').read_bytes()).hexdigest(),
      'externalWeights':False,'personalChats':False,'oldClosedBudgetsRestarted':False}
if not a.resume:
    (root/'varied-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
else:
    original=json.loads((root/'varied-plan.json').read_text())
    assert original['corpusSha256']==plan['corpusSha256']
    assert original['request']['training_config']==config
status=train(**request)
print(json.dumps({'segment':a.segment,'status':status},ensure_ascii=False),flush=True)
best=export_best(run,root/f'varied-s{a.segment}-best')
model,tokenizer,state,_=_load_checkpoint(run,status['lastCheckpoint'])
metadata={'corpusSha256':state['corpus_sha256'],'selectedStep':state['step'],
          'selectionSplit':'declared_segment_endpoint','config':state['training_config'],
          'assistantTargets':state['assistant_targets'],**_origin_metadata(state),**_loss_metadata(state['training_config'])}
last=save_artifact(root/f'varied-s{a.segment}-last',model,tokenizer,training=metadata)
summary={'segment':a.segment,'status':status,'bestModelId':best['modelId'],
         'bestStep':best['training']['selectedStep'],'lastModelId':last['modelId'],'lastStep':state['step']}
(root/f'varied-s{a.segment}-result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False),flush=True)
