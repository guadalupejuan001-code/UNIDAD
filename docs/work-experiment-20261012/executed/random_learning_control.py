"""Matched random-weight control requested by the updated repository orders."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from ia_propia.conversation.artifacts import save_artifact
from ia_propia.conversation.model import ConversationModel
from ia_propia.conversation.training import train, export_best

root=Path(__file__).resolve().parent
base=root.parent/'UNIDAD/models/conversation-reference-c7'
torch.set_num_threads(4)
config=json.loads((base/'config.json').read_text())
tokenizer=Tokenizer.from_file(str(base/'tokenizer.json'))
model=ConversationModel(config)
corpus=root/'learning-control.jsonl'
manifest=save_artifact(root/'random-control-initial',model,tokenizer,training={
    'trainedFrom':'random_weights','weightsTrainingSteps':0,
    'corpusSha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
    'tokenizerOrigin':{'ownTokenizerReused':True,'sameAsC7':True,'tokenizerSha256':hashlib.sha256((base/'tokenizer.json').read_bytes()).hexdigest()},
    'purpose':'Comparable control: random tensors, same architecture, vocabulary, corpus and training settings as the C7 memorization control'})
prior=json.loads((root/'control-plan.json').read_text())
request={**prior['request'],'run_dir':str(root/'random-control-run'),'initial_model':str(root/'random-control-initial')}
plan={'reason':'Updated ORDENES_USUARIO.md in a1a283ca requests comparable learning controls from random weights.',
      'request':request,'randomInitialModelId':manifest['modelId'],'matchedTo':'control-plan.json',
      'differingFactor':'initial weights only; tokenizer remains the same own tokenizer',
      'memorizationOnly':True,'testUsed':False}
(root/'random-control-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
status=train(**request)
best=export_best(root/'random-control-run',root/'random-control-model')
print(json.dumps({'status':status,'modelId':best['modelId'],'selectedStep':best['training']['selectedStep'],'validationNll':best['training']['validationNll']}),flush=True)
torch.set_num_threads(1)
from ia_propia.conversation.runtime import ConversationEngine
engine=ConversationEngine(root/'random-control-model')
results=[]
for record in [json.loads(l) for l in corpus.read_text().splitlines()]:
    if record['split']!='train':continue
    actual=engine.reply(record['messages'][:-1],max_new_tokens=32,temperature=0,seed=42)
    results.append({'id':record['id'],'messages':record['messages'][:-1],'target':record['messages'][-1]['content'],
                    'actual':actual,'exactMatch':actual['text']==record['messages'][-1]['content']})
report={'purpose':'TRAIN-only memorization, not generalization','modelId':engine.model_id,
        'selectedStep':best['training']['selectedStep'],'count':len(results),
        'exactMatches':sum(r['exactMatch'] for r in results),'records':results}
(root/'random-control-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['count','exactMatches','modelId','selectedStep']}),flush=True)
