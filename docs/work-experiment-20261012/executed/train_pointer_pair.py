"""Matched fixed-budget v1/v2 source-pointer experiment, using native training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from ia_propia.conversation.artifacts import load_artifact,save_artifact
from ia_propia.conversation.model import ConversationModel
from ia_propia.conversation.training import train,export_best

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=['v1','v2'],required=True)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parent
torch.set_num_threads(4)
base=root.parent/'UNIDAD/models/conversation-reference-c7'
if a.arm=='v2':
    source=root/'source-pointer-v2-initial'
    if not source.exists():
        old,tokenizer,parent=load_artifact(base)
        config=old.config;config['copy_head']['version']=2
        model=ConversationModel(config)
        model.load_state_dict(old.state_dict(),strict=True)
        unchanged=all(torch.equal(v.view(torch.uint8),model.state_dict()[k].view(torch.uint8)) for k,v in old.state_dict().items())
        if not unchanged:raise RuntimeError('Conversion must not modify inherited tensors')
        lineage={'parentModelId':parent['modelId'],'parentWeightsSha256':parent['files']['weights.safetensors']['sha256'],'parentCorpusSha256':parent['training']['corpusSha256'],'rootOrigin':'random_weights'}
        save_artifact(source,model,tokenizer,training={'trainedFrom':'own_checkpoint','lineage':lineage,'corpusSha256':parent['training']['corpusSha256'],'intervention':'copy_head version 2 uses h[j] to copy source token j','inheritedTensorsByteExact':True,'trainingStepsSinceParent':0})
        del model,old
else:source=base
config={'batch_size':8,'learning_rate':0.0001,'weight_decay':0.01,'grad_clip':1.0,
        'max_epochs':1000,'max_steps':256,'max_seconds':600,'validation_every':128,
        'checkpoint_every':128,'patience':12,'seed':61,'torch_threads':4,'warmup_steps':32,
        'window_policy':'sliding_context','objective':'assistant_last_reply','loss_reduction':'token_mean',
        'projection_mode':'supervised','partition_policy':'train_validation_only','train_precision':'cpu_bfloat16',
        'copy_supervision':{'alignment_weight':0.2,'gate_weight':0.2}}
run=root/f'pointer-{a.arm}-run'
request={'corpus_path':str(root/'repair-corpus.jsonl'),'run_dir':str(run),
         'model_config':json.loads((source/'config.json').read_text()),'training_config':config,
         'resume':a.resume,'budget_seconds':600}
if not a.resume:request['initial_model']=str(source)
plan={'schemaVersion':1,'arm':a.arm,'frozenAtUnix':time.time(),'request':request,
      'hypothesis':'A key including the actual source token makes copying and disambiguation easier than a key containing only its preceding context. This is experimental, not a proven explanation of all C7 failures.',
      'changedBetweenArms':['copy_head.version; keys h[j-1] versus h[j]'],
      'matched':['all starting tensor bytes','tokenizer','corpus','split','training seed','optimizer initialization','configuration','fixed endpoint 256 steps','generation'],
      'selection':'Report both fixed endpoints for architecture comparison; export validation-best independently. No generation-based checkpoint selection.',
      'knownTarget':'F <= 14/40, critical failures <= 6; report complete and partial responses too',
      'sourceCommit':'b5fc677dfeb90b5e9cbd50f26cf878fad6a79d6a',
      'hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['repair-corpus.jsonl','known-dev.json','fresh-probe.json']},
      'testUsedForTrainingOrSelection':False,'externalWeights':False,'runtimeTemplates':False,
      'userScreenshotConsidered':'2026-10-12 13:18 America/New_York: other Codex already ran revised-text experiments; this pair isolates source-key alignment.'}
if not a.resume:(root/f'pointer-{a.arm}-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=train(**request)
print(json.dumps(status,ensure_ascii=False),flush=True)
best=export_best(run,root/f'pointer-{a.arm}-best')
print(json.dumps({'bestModelId':best['modelId'],'selectedStep':best['training']['selectedStep'],'validationNll':best['training']['validationNll']}),flush=True)
# Also export the fixed endpoint; this does not choose by observed generation.
from ia_propia.conversation.training import _load_checkpoint,_origin_metadata,_loss_metadata
model,tokenizer,state,_=_load_checkpoint(run,status['lastCheckpoint'])
metadata={'corpusSha256':state['corpus_sha256'],'selectedStep':state['step'],'selectionSplit':'prospective_fixed_endpoint','config':state['training_config'],'assistantTargets':state['assistant_targets'],**_origin_metadata(state),**_loss_metadata(state['training_config'])}
endpoint=save_artifact(root/f'pointer-{a.arm}-endpoint',model,tokenizer,training=metadata)
print(json.dumps({'endpointModelId':endpoint['modelId'],'step':state['step']}),flush=True)
