"""Check memorization on TRAIN examples; not a conversation acceptance score."""
import json
from pathlib import Path
import torch
from ia_propia.conversation.runtime import ConversationEngine

root=Path(__file__).resolve().parent
torch.set_num_threads(1)
records=[json.loads(l) for l in (root/'learning-control.jsonl').read_text().splitlines()]
records=[r for r in records if r['split']=='train']
result={'purpose':'Small TRAIN-only memorization check, with authored earlier assistant history; NOT generalization or actual-history conversation acceptance','records':[]}
for label,path in [('baseline',root.parent/'UNIDAD/models/conversation-reference-c7'),('control',root/'control-model')]:
    engine=ConversationEngine(path)
    for r in records:
        actual=engine.reply(r['messages'][:-1],max_new_tokens=32,temperature=0,seed=42)
        result['records'].append({'model':label,'modelId':engine.model_id,'recordId':r['id'],'prompt':r['messages'][:-1],'target':r['messages'][-1]['content'],'actual':actual,'exactMatch':actual['text']==r['messages'][-1]['content']})
    print(label,sum(x['exactMatch'] for x in result['records'] if x['model']==label),'/',len(records),flush=True)
result['exactMatches']={label:sum(x['exactMatch'] for x in result['records'] if x['model']==label) for label in ('baseline','control')}
(root/'learning-control-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
