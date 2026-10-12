"""Bounded, native and resumable continuation from unchanged C7 own weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from ia_propia.conversation.training import train, export_best

p = argparse.ArgumentParser()
p.add_argument('--control',action='store_true')
p.add_argument('--resume',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parent
control=a.control
config={'batch_size':8,'learning_rate':0.0005 if control else 0.0001,'weight_decay':0.01,'grad_clip':1.0,
        'max_epochs':1000,'max_steps':64 if control else 768,'max_seconds':240 if control else 1200,
        'validation_every':32 if control else 128,'checkpoint_every':32 if control else 128,
        'patience':12,'seed':61,'torch_threads':4,'warmup_steps':16 if control else 32,
        'window_policy':'sliding_context','objective':'assistant_last_reply',
        'loss_reduction':'token_mean','projection_mode':'supervised',
        'partition_policy':'train_validation_only','train_precision':'cpu_bfloat16',
        'copy_supervision':{'alignment_weight':0.2,'gate_weight':0.2}}
run=root/('control-run' if control else 'repair-run')
export=root/('control-model' if control else 'repair-model')
corpus=root/('learning-control.jsonl' if control else 'repair-corpus.jsonl')
request={'corpus_path':str(corpus),'run_dir':str(run),'training_config':config,
         'resume':a.resume,'budget_seconds':config['max_seconds']}
request['model_config']=json.loads((root.parent/'UNIDAD/models/conversation-reference-c7/config.json').read_text())
if not a.resume: request['initial_model']=str(root.parent/'UNIDAD/models/conversation-reference-c7')
record={'schemaVersion':1,'createdAtUnix':time.time(),'request':request,
        'purpose':'Memorization control only' if control else 'Prospective bounded repair; validation NLL selects checkpoint',
        'hypothesis':'Balanced multi-turn supervision with some neutral past assistant replies improves reliance on user facts and corrections. Preserve C7 copy head, copy auxiliary coefficients, token-mean loss and production inference. This is a data intervention; it does not isolate individual curriculum ingredients.',
        'baselineWeightsSha256':'666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836',
        'corpusSha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'knownDevSha256':hashlib.sha256((root/'known-dev.json').read_bytes()).hexdigest(),
        'freshProbeSha256':hashlib.sha256((root/'fresh-probe.json').read_bytes()).hexdigest(),
        'successCriterion':'Known development F <= 14 of 40, critical failures no greater than 6; disclose C/P/F and fresh transfer separately',
        'freshProbeUsedForSelection':False,'optimizerReset':not a.resume,'schedulerReset':not a.resume,
        'externalModelUsed':False,'runtimeTemplates':False}
record_path=root/('control-plan.json' if control else 'repair-plan.json')
if not a.resume: record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=train(**request)
print(json.dumps(status,ensure_ascii=False),flush=True)
artifact=export_best(run,export)
print(json.dumps({'exportedModelId':artifact['modelId'],'selectedStep':artifact['training']['selectedStep'],'validationNll':artifact['training']['validationNll']},ensure_ascii=False),flush=True)
