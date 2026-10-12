"""Prospective bounded search comparison on already exposed diagnostic pairs."""
import json
import time
from pathlib import Path
import torch
from ia_propia.conversation.runtime import ConversationEngine

root=Path(__file__).resolve().parent
old=root.parent/'unidad-fix'
pairs=json.loads((old/'wording-probe.json').read_text())['pairs']
plan={'createdAtUnix':time.time(),'changed':'deterministic beam width 1 versus 8; same native engine and fixed weights',
      'hypothesis':'Some wrong early greedy choices may lose a higher probability complete reply; no expected reply enters search.',
      'lengthNormalization':False,'repetitionRules':False,'training':False,
      'generation':{'temperature':0,'seed':42,'maxNewTokens':32},
      'scope':'Already exposed diagnostic development, not final acceptance',
      'gate':'If beam shows a semantic benefit on these pairs, compare both models on the same known 40 turns before adopting anything.'}
(root/'beam-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
torch.set_num_threads(1)
out={'plan':plan,'models':{}}
for label,path in [('C7',root.parent/'UNIDAD/models/conversation-reference-c7'),('adjusted',old/'pointer-v1-best')]:
    engine=ConversationEngine(path)
    data={'modelId':engine.model_id,'pairs':[]}
    for pair in pairs:
        result={'family':pair['family'],'target':pair['target'],'user':pair['paraphrase'],'responses':{}}
        for width in [1,8]:
            started=time.monotonic()
            actual=engine.reply([{'role':'user','content':pair['paraphrase']}],max_new_tokens=32,temperature=0,seed=42,beam_width=width)
            result['responses'][str(width)]={'actual':actual,'elapsedSeconds':time.monotonic()-started,
                                             'exact':actual['text']==pair['target']}
        data['pairs'].append(result)
        out['models'][label]=data
        (root/'beam-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'model':label,'family':pair['family'],'responses':{w:r['actual']['text'] for w,r in result['responses'].items()}},ensure_ascii=False),flush=True)
    print(json.dumps({'model':label,'exact':{w:sum(p['responses'][w]['exact'] for p in data['pairs']) for w in ['1','8']}}),flush=True)
