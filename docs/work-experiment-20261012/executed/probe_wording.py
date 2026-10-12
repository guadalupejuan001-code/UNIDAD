"""Post-hoc wording diagnostic on existing validation worlds, never model selection."""
import json
from pathlib import Path
import re
import torch
from ia_propia.conversation.runtime import ConversationEngine

root=Path(__file__).resolve().parent
records=[json.loads(l) for l in (root/'repair-corpus.jsonl').read_text().splitlines()]
records={r['source']['family']:r for r in records if r['groupId']=='repair-20261012-world-088' and r['source']['prefixPairs']==1}
pairs=[]
def add(family, changed):
    r=records[family]
    pairs.append({'family':family,'recordId':r['id'],'original':r['messages'][0]['content'],
                  'paraphrase':changed,'target':r['messages'][-1]['content']})
r=records['preferred_address']['messages'][0]['content']
name,alias=re.search(r'Mi nombre es (.*?), pero uso el apodo (.*?)\.',r).groups()
add('preferred_address',f'Para dirigirte a mí usa {alias}. Mi nombre completo es {name}. ¿Qué apodo he escogido?')
add('missing_information','Aún no te he contado el color de mi abrigo. ¿Tienes información para decir de qué color es?')
r=records['current_and_future_place']['messages'][0]['content']
city,dest=re.search(r'Hoy sigo en (.*?)\. Después viajaré a (.*?)\.',r).groups()
add('current_and_future_place',f'Tengo planeado un viaje a {dest}, pero sigo en {city}. ¿Cuál es mi ubicación de hoy?')
add('listening','Necesito desahogarme de un día difícil. Acompáñame escuchando y deja los consejos.')
r=records['project_change']['messages'][0]['content']
title=re.search(r'El proyecto (.*?) reúne dibujos\.',r).group(1)
add('project_change',f'Estoy reuniendo dibujos en {title}. Dime cómo se llama mi proyecto y qué contiene.')
r=records['reschedule']['messages'][0]['content']
name,day,hour=re.search(r'Quedé con (.*?) el (.*?) a (.*?)\.',r).groups()
add('reschedule',f'Mi encuentro con {name} será el {day}, a {hour}. Repite solamente el día y la hora.')
r=records['two_objects']['messages'][0]['content']
color,other=re.search(r'Mi bolso es (.*?) y mi reloj es (.*?)\.',r).groups()
add('two_objects',f'El reloj es {other}; el bolso, {color}. Recuérdame el color del reloj.')
r=records['unknown_author']['messages'][0]['content']
title=re.search(r'Quiero encontrar una novela llamada (.*?)\.',r).group(1)
add('unknown_author',f'Necesito una novela: {title}. ¿Cuál es el título de esa novela?')
torch.set_num_threads(1)
engine=ConversationEngine(root/'pointer-v1-best')
for p in pairs:
    for kind in ['original','paraphrase']:
        p[kind+'Actual']=engine.reply([{'role':'user','content':p[kind]}],max_new_tokens=32,temperature=0,seed=42)
        p[kind+'Exact']=p[kind+'Actual']['text']==p['target']
    print(p['family'],p['originalActual']['text'],'=>',p['paraphraseActual']['text'],flush=True)
result={'purpose':'Post-hoc wording sensitivity on a validation world, not final evaluation or a training source',
        'modelId':engine.model_id,'pairs':pairs,'exactMatches':{kind:sum(p[kind+'Exact'] for p in pairs) for kind in ['original','paraphrase']},
        'count':len(pairs),'earlierAssistantHistoryUsed':False,'weightsChanged':False,
        'limitation':'Exact-match is a diagnostic, not semantic correctness; paraphrases of the expected answer may be valid.'}
(root/'wording-probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result['exactMatches'],flush=True)
