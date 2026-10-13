"""Real generated history on declared development; fresh transfer is separate."""
import argparse
import json
from pathlib import Path
import sys
import torch
from ia_propia.conversation.runtime import ConversationEngine

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'unidad-fix'))
from evaluate import evaluate

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--label',required=True)
p.add_argument('--fresh',action='store_true')
a=p.parse_args()
if a.fresh:
    assert (root/'candidate-selection.json').exists(), 'Freeze candidate choice before exposing fresh generations'
    evaluate(a.model,root/'fresh-transfer.json',root/f'{a.label}.fresh.actual.json')
    raise SystemExit(0)
evaluate(a.model,root.parent/'unidad-fix/known-dev.json',root/f'{a.label}.known.actual.json')
evaluate(a.model,root/'wording-validation.json',root/f'{a.label}.validation.actual.json')
torch.set_num_threads(1)
engine=ConversationEngine(a.model)
pairs=json.loads((root.parent/'unidad-fix/wording-probe.json').read_text())['pairs']
out={'modelId':engine.model_id,'purpose':'Already exposed wording diagnostic, not final acceptance','pairs':[]}
for pair in pairs:
    actual=engine.reply([{'role':'user','content':pair['paraphrase']}],max_new_tokens=32,temperature=0,seed=42)
    out['pairs'].append({'family':pair['family'],'user':pair['paraphrase'],'expected':pair['target'],'actual':actual,'exact':actual['text']==pair['target']})
out['exact']=sum(x['exact'] for x in out['pairs'])
(root/f'{a.label}.wording.actual.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
suite=json.loads((root/'wording-validation.json').read_text())
targets={t['id']:t['expectedMeaning'] for e in suite['episodes'] for t in e['turns']}
validation=json.loads((root/f'{a.label}.validation.actual.json').read_text())
summary={'modelId':engine.model_id,'label':a.label,'wordingExact':out['exact'],
         'validationExact':sum(t['actual']['text']==targets[t['id']] for e in validation['episodes'] for t in e['turns']),
         'validationTotal':40,'semanticReviewRequired':True}
(root/f'{a.label}.generation-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False),flush=True)
