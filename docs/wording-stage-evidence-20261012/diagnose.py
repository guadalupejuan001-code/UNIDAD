"""Read-only paired diagnosis, using own saved models and native generation."""
import hashlib
import json
from pathlib import Path
import time

import torch
from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import encode_prompt
from ia_propia.conversation.contract import EOS_ID, PAD_ID, BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'unidad-fix'
torch.set_num_threads(1)
pairs = json.loads((OLD / 'wording-probe.json').read_text())['pairs']
records = [json.loads(line) for line in (OLD / 'repair-corpus.jsonl').read_text().splitlines()]
full = [r for r in records if r['groupId'] == 'repair-20261012-world-088' and r['source']['prefixPairs'] == 4]
models = {'C7': ROOT.parent / 'UNIDAD/models/conversation-reference-c7',
          'adjusted': OLD / 'pointer-v1-best'}

def detail(engine, messages, target):
    model, tokenizer = engine._model, engine._tokenizer
    ids, discarded = encode_prompt(messages, tokenizer, model.config['context_length'])
    targets = tokenizer.encode(target, add_special_tokens=False).ids + [EOS_ID]
    inputs = torch.tensor([ids + targets[:-1]], dtype=torch.long)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[:, len(ids)-1:] = True
    model.eval()
    with torch.inference_mode():
        scores, info = model._copy_project(inputs, mask, details=True)
        logits = scores.float().clone()
        logits[:, [PAD_ID, BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID]] = -torch.inf
        prediction = logits.argmax(-1)
        labels = torch.tensor(targets)
        probabilities = torch.softmax(logits, -1)
        chosen = probabilities.gather(1, labels[:, None]).flatten()
        correct = prediction == labels
        failures = (~correct).nonzero().flatten().tolist()
        details = []
        for i in failures[:4]:
            top_scores, top_ids = probabilities[i].topk(5)
            attention, source_columns = info['attention'][i].topk(min(3, info['attention'].shape[1]))
            details.append({'targetIndex':i, 'expectedToken':tokenizer.id_to_token(targets[i]),
                            'expectedDecoded':tokenizer.decode([targets[i]],skip_special_tokens=False),
                            'expectedProbability':float(chosen[i]),
                            'expectedRank':int((probabilities[i] > chosen[i]).sum())+1,
                            'copyGate':float(info['gate'][i]),
                            'top': [{'token':tokenizer.id_to_token(int(t)), 'probability':float(p)} for t,p in zip(top_ids,top_scores)],
                            'copySources':[{'token':tokenizer.id_to_token(int(info['sourceIds'][i, col])),
                                            'position':int(col)+1,'attention':float(p)} for col,p in zip(source_columns,attention)]})
    return {'promptTokens':len(ids),'discarded':discarded,'targetTokens':len(targets),
            'correctWithIdealPrefix':int(correct.sum()),'allCorrectWithIdealPrefix':bool(correct.all()),
            'meanNll':float(-chosen.clamp_min(1e-38).log().mean()),
            'meanCopyGate':float(info['gate'].mean()),'firstErrors':details}

plan = {'stage':'principles-diagnosis','createdAtUnix':time.time(),
        'sourceCommit':'a1a283caabf89b3191c8ce64c2a950e9edbb0edf',
        'purpose':'Distinguish wording sensitivity, ideal-prefix prediction and effects of earlier assistant replies before choosing another intervention.',
        'weightsChanged':False,'trainingPerformed':False,'newBudgetNotResumeOfClosedRuns':True,
        'generation':{'temperature':0,'seed':42,'maxNewTokens':32,'torchThreads':1},
        'comparison':['same facts, original versus paraphrased first turn','same user sequence, authored versus actually generated assistant history'],
        'exactMatchLimitation':'Exact equality is only a diagnostic; semantic review remains necessary.',
        'previousCorpusSha256':hashlib.sha256((OLD/'repair-corpus.jsonl').read_bytes()).hexdigest()}
(ROOT/'diagnostic-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
results={'plan':plan,'models':{}}
for label, path in models.items():
    engine=ConversationEngine(path)
    data={'modelId':engine.model_id,'wording':[], 'history':[]}
    results['models'][label]=data
    for pair in pairs:
        result={'family':pair['family'],'target':pair['target'],'variants':{}}
        for kind in ['original','paraphrase']:
            messages=[{'role':'user','content':pair[kind]}]
            actual=engine.reply(messages,max_new_tokens=32,temperature=0,seed=42)
            result['variants'][kind]={'user':pair[kind],'actual':actual,
                                     'exact':actual['text']==pair['target'],
                                     'targetDiagnostic':detail(engine,messages,pair['target'])}
        data['wording'].append(result)
        print(json.dumps({'model':label,'family':pair['family'],
                          'outputs':{k:v['actual']['text'] for k,v in result['variants'].items()}},ensure_ascii=False),flush=True)
    for record in full:
        family_records = sorted([r for r in records if r['groupId'] == record['groupId']
                                 and r['source']['family'] == record['source']['family']],
                                key=lambda r:r['source']['prefixPairs'])
        # Each prefix retains its own true target even when older replies were
        # neutralized. Reconstruct authored history from those explicit targets.
        assert [r['source']['prefixPairs'] for r in family_records] == [1,2,3,4]
        source=[dict(m) for r in family_records for m in r['messages'][-2:]]
        trial={'family':record['source']['family'],'modes':{}}
        for mode in ['authored','actual']:
            history=[]; turns=[]
            for j in range(0,len(source),2):
                user,target=source[j],source[j+1]
                history.append(user)
                actual=engine.reply(history,max_new_tokens=32,temperature=0,seed=42)
                turns.append({'user':user['content'],'expected':target['content'],
                              'inputMessages':[dict(m) for m in history],
                              'actual':actual,'exact':actual['text']==target['content'],
                              'targetDiagnostic':detail(engine,history,target['content'])})
                history.append(target if mode=='authored' else {'role':'assistant','content':actual['text']})
            trial['modes'][mode]=turns
        data['history'].append(trial)
        (ROOT/'diagnosis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'model':label,'family':trial['family'],
                          'exactByHistory':{m:sum(t['exact'] for t in ts) for m,ts in trial['modes'].items()}},ensure_ascii=False),flush=True)
    data['summary']={'wordingExact':{k:sum(p['variants'][k]['exact'] for p in data['wording']) for k in ['original','paraphrase']},
                     'wordingNll':{k:sum(p['variants'][k]['targetDiagnostic']['meanNll'] for p in data['wording'])/len(pairs) for k in ['original','paraphrase']},
                     'historyExact':{k:sum(t['exact'] for family in data['history'] for t in family['modes'][k]) for k in ['authored','actual']}}
    results['models'][label]=data
    (ROOT/'diagnosis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'model':label,'summary':data['summary']},ensure_ascii=False),flush=True)
    del engine
