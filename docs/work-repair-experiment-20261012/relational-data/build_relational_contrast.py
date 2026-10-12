"""Bounded four-family TRAIN-only state-map contrast. Never trains or edits core.

Preserve original slots/IDs/groups/folds. Copy eight other TRAIN families and
every original VAL line byte-for-byte. No diagnostic/VAL prompts author TRAIN.
Only order, requested binding, explicit correction and gold vary in four tasks.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata

os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path('/workspace/unidad-conversation-repair')
BASE=ROOT/'private/pr1-relational-data-contrast-v1'
SOURCEBASE=ROOT/'private/pr1-wording-composition-audit-v1'
sys.path.insert(0,str(ROOT/'src'))
from tokenizers import Tokenizer
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, encode_dialogue

EXPECTED='862996c131ea17e3623f55530b2f6d61e1d4f64e37672bc99581c19df17a7186'
VALSHA='e305adb7fec7c045d2b3ca2d066297aac9dcc04bc5f580c54dae55ac664ab37c'
FAMILIES={'preferred_address','current_and_future_place','two_people','two_objects'}
COLORS=['azul','verde','gris','amarillo','negro','blanco','rojo','violeta']
FIELDS=['name','other','alias','new_alias','city','dest','changed','day','day2','hour','place','first','second','color','color2','title','title2']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
norm=lambda s:' '.join(unicodedata.normalize('NFC',s).casefold().split())

def skeleton(text,w):
    values=[(w[k],'<'+k.upper()+'>') for k in FIELDS]
    if 'correction_color' in w:values.append((w['correction_color'],'<CORRECTION_COLOR>'))
    for value,label in sorted(values,key=lambda x:-len(x[0])):text=text.replace(value,label)
    return norm(text)

def preferred(w,o,q,u):
    name,old,new=w['name'],w['alias'],w['new_alias'];style=w['style']
    state={'fullName':name,'preferredAddress':old,'previousPreferredAddress':None}
    first=[f'Soy {name}; quiero que me digas {old}.',f'Mi nombre es {name}, pero uso el apodo {old}.',f'Me llamo {name}, aunque puedes llamarme {old}.'][style]
    if o:first=[f'Quiero que me digas {old}; soy {name}.',f'Uso el apodo {old}, pero mi nombre es {name}.',f'Puedes llamarme {old}, aunque me llamo {name}.'][style]
    addressq=['¿Qué nombre usarás conmigo?','¿Cómo me llamarás?','Repite mi preferencia.'][style]
    turns=[];trace=[]
    user=first+' '+('¿Cuál es mi nombre?' if q else addressq)
    gold=f'Te llamas {state["fullName"]}.' if q else f'Te llamaré {state["preferredAddress"]}.'
    turns.append((user,gold));trace.append({'state':deepcopy(state),'requested':'identification.fullName' if q else 'address.preferredAddress'})
    state['desiredActivity']=w['first']
    turns.append((f'Esta tarde quiero {w["first"]}. ¿Qué quiero hacer?',f'Quieres {state["desiredActivity"]}.'))
    trace.append({'state':deepcopy(state),'requested':'desiredActivity'})
    before=deepcopy(state);state['previousPreferredAddress']=state['preferredAddress'];state['preferredAddress']=new
    intro=[f'He cambiado de idea: prefiero {new}.',f'Desde ahora dime {new}.',f'Corrijo mi preferencia: usa {new}.'][style]
    currentq=['¿Qué apodo usarás ahora?','¿Cómo vas a llamarme?','¿Cuál es mi apodo actual?'][style]
    identify=not q
    turns.append((intro+' '+('¿Cuál es mi nombre?' if identify else currentq),f'Te llamas {state["fullName"]}.' if identify else f'Ahora te llamaré {state["preferredAddress"]}.'))
    assert state['fullName']==before['fullName'] and state['desiredActivity']==before['desiredActivity']
    trace.append({'state':deepcopy(state),'requested':'identification.fullName' if identify else 'address.preferredAddress','updateOnly':'preferredAddress; preserve fullName and desiredActivity'})
    turns.append(('Distingue el apodo anterior y el que elegí después.',f'Antes preferías {state["previousPreferredAddress"]}; ahora, {state["preferredAddress"]}.'))
    trace.append({'state':deepcopy(state),'requested':'previousPreferredAddress,currentPreferredAddress'})
    return turns,trace,{'presentationOrder':'alias-first' if o else 'name-first','firstRequestedRole':'identification' if q else 'preferred-address','thirdRequestedRole':'preferred-address' if q else 'identification','unusedCellBit':u}

def places(w,o,q,u):
    state={'currentPlace':w['city'],'futurePlace':w['dest']};style=w['style']
    intros=[f'Ahora estoy en {state["currentPlace"]}; el próximo viaje será a {state["futurePlace"]}.',f'Hoy sigo en {state["currentPlace"]}. Después viajaré a {state["futurePlace"]}.',f'Mi ubicación es {state["currentPlace"]} y mi destino es {state["futurePlace"]}.']
    reverse=[f'El próximo viaje será a {state["futurePlace"]}; ahora estoy en {state["currentPlace"]}.',f'Después viajaré a {state["futurePlace"]}. Hoy sigo en {state["currentPlace"]}.',f'Mi destino es {state["futurePlace"]} y mi ubicación es {state["currentPlace"]}.']
    currentq=['¿Dónde estoy?','¿Cuál es mi ubicación actual?','¿En qué lugar estoy ahora?'][style]
    futureq=['¿A dónde viajaré?','¿Cuál es mi destino?','¿A qué lugar viajaré?'][style]
    turns=[((reverse if o else intros)[style]+' '+(futureq if q else currentq),f'Irás a {state["futurePlace"]}.' if q else f'Estás en {state["currentPlace"]}.')]
    trace=[{'state':deepcopy(state),'requested':'futurePlace' if q else 'currentPlace'}]
    before=deepcopy(state);state['futurePlace']=w['changed']
    user=f'Finalmente iré a {state["futurePlace"]} en lugar de {before["futurePlace"]}. '+('¿Cuál es mi ubicación actual?' if u else '¿Cuál es el nuevo destino?')
    turns.append((user,f'Estás en {state["currentPlace"]}.' if u else f'Irás a {state["futurePlace"]}.'))
    assert state['currentPlace']==before['currentPlace']
    trace.append({'state':deepcopy(state),'requested':'currentPlace' if u else 'futurePlace','updateOnly':'futurePlace; preserve currentPlace'})
    asked=state['currentPlace'] if q else state['futurePlace'];yes=asked==state['currentPlace']
    turns.append((f'No he salido de {state["currentPlace"]}. ¿Estoy ya en {asked}?',f'{"Sí" if yes else "No"}, sigues en {state["currentPlace"]}.'))
    trace.append({'state':deepcopy(state),'requested':{'isCurrentlyAt':asked,'answer':yes}})
    turns.append(('Dime desde dónde salgo y hacia dónde viajaré.',f'Saldrás de {state["currentPlace"]} hacia {state["futurePlace"]}.'))
    trace.append({'state':deepcopy(state),'requested':'currentPlace,futurePlace'})
    return turns,trace,{'presentationOrder':'future-first' if o else 'current-first','firstRequestedRole':'future' if q else 'current','secondRequestedRole':'current' if u else 'future','thirdYesNo':yes}

def people(w,o,q,u):
    names=[w['name'],w['other']];order=[1,0] if o else [0,1]
    state={names[0]:{'relation':'hermana','sings':None,'crop':None},names[1]:{'relation':'vecino','sings':None,'crop':None}}
    clauses={0:f'{names[0]} es mi hermana',1:f'{names[1]} es mi vecino'}
    turns=[(clauses[order[0]]+' y '+clauses[order[1]]+'. Dime quién es cada uno.',f'{names[order[0]]} es tu {state[names[order[0]]]["relation"]}; {names[order[1]]}, tu {state[names[order[1]]]["relation"]}.')]
    trace=[{'state':deepcopy(state),'requested':'both relations in stated order'}]
    singer=q;gardener=1-q;state[names[singer]]['sings']=True;state[names[gardener]]['crop']='rosas'
    clauses={singer:f'{names[singer]} canta',gardener:f'{names[gardener]} cultiva rosas'}
    turns.append((clauses[order[0]]+' y '+clauses[order[1]]+'. ¿Quién canta?',f'Canta {names[singer]}.'))
    trace.append({'state':deepcopy(state),'requested':'singing actor'})
    before=deepcopy(state);state[names[u]]['crop']='rosas';state[names[1-u]]['crop']='tulipanes'
    flower='tulipanes' if q else 'rosas';actor=next(n for n in names if state[n]['crop']==flower)
    def cropclause(i):return f'{names[i]} cultiva '+('las rosas' if state[names[i]]['crop']=='rosas' else 'tulipanes')
    turns.append(('Corrijo: '+cropclause(order[0])+'; '+cropclause(order[1])+f'. ¿Quién cuida {"los tulipanes" if q else "las rosas"}?',f'{actor} cuida {"los tulipanes" if q else "las rosas"}.'))
    for n in names:assert state[n]['relation']==before[n]['relation'] and state[n]['sings']==before[n]['sings']
    trace.append({'state':deepcopy(state),'requested':{'cropActor':flower},'updateOnly':'crop assignments; preserve relationships and singing facts'})
    relationperson=names[q];cropperson=names[q^u]
    turns.append((f'¿Quién es {relationperson} para mí y qué cultiva {cropperson}?',f'{relationperson} es tu {state[relationperson]["relation"]}; {cropperson} cultiva {state[cropperson]["crop"]}.'))
    trace.append({'state':deepcopy(state),'requested':{'relationPerson':relationperson,'cropPerson':cropperson}})
    return turns,trace,{'presentationOrder':'B-first' if o else 'A-first','singer':'B' if q else 'A','rosesActorAfterCorrection':'B' if u else 'A','thirdRequestedCrop':flower,'lastRelationPerson':'B' if q else 'A','lastCropPerson':'B' if q^u else 'A'}

def objects(w,o,q,u,equal):
    objects=['bolso','reloj'];order=[1,0] if o else [0,1]
    state={'bolso':{'color':w['color'],'giver':None},'reloj':{'color':w['color2'],'giver':None}}
    def colorof(i):return state[objects[i]]['color']
    target=objects[q]
    turns=[(f'Mi {objects[order[0]]} es {colorof(order[0])} y mi {objects[order[1]]} es {colorof(order[1])}. ¿De qué color es el {target}?',f'El {target} es {state[target]["color"]}.')]
    trace=[{'state':deepcopy(state),'requested':{'colorOf':target}}]
    state[target]['giver']=w['name']
    turns.append((f'El {target} me lo regaló {w["name"]}. ¿Quién me lo regaló?',f'Te lo regaló {state[target]["giver"]}.'))
    trace.append({'state':deepcopy(state),'requested':{'giverOf':target}})
    before=deepcopy(state);updated=objects[u];untouched=objects[1-u]
    state[updated]['color']=before[untouched]['color'] if equal else w['correction_color']
    turns.append((f'Antes me equivoqué: el {updated} es {state[updated]["color"]}. ¿Cuál es el color correcto del {target}?',f'El {target} es {state[target]["color"]}.'))
    assert state[untouched]==before[untouched] and state[updated]['giver']==before[updated]['giver']
    assert (state['bolso']['color']==state['reloj']['color']) is equal
    assert state[updated]['color']!=before[updated]['color']
    trace.append({'state':deepcopy(state),'requested':{'colorOf':target},'updateOnly':f'{updated}.color; preserve other color and both givers'})
    giftquery=updated;giver=state[giftquery]['giver']
    gold=f'Te lo regaló {giver}.' if giver is not None else f'No has dicho quién te dio el {giftquery}.'
    turns.append((f'¿Qué sabemos de quién me dio el {giftquery}?',gold))
    trace.append({'state':deepcopy(state),'requested':{'giverOf':giftquery,'known':giver is not None}})
    return turns,trace,{'presentationOrder':'watch-first' if o else 'bag-first','requestedColorEntity':target,'giverDisclosedFor':target,'correctedColorEntity':updated,'thirdQueriesCorrectedEntity':q==u,'lastQueriesKnownGiver':q==u,'colorsEqualAfterCorrection':equal,'newCorrectionColor':state[updated]['color']}

COMPILERS={'preferred_address':preferred,'current_and_future_place':places,'two_people':people,'two_objects':objects}

def main():
    sourcepath=SOURCEBASE/'reproduced-original/repair-corpus.jsonl';assert sha(sourcepath)==EXPECTED
    sourcebytes=sourcepath.read_bytes().splitlines(keepends=True);old=[json.loads(x) for x in sourcebytes]
    worlds=json.loads((SOURCEBASE/'reproduced-world-metadata.json').read_text())
    valpath=SOURCEBASE/'prospective-validation/validation.jsonl';assert sha(valpath)==VALSHA
    newvalbytes=valpath.read_bytes();newval=[json.loads(x) for x in newvalbytes.splitlines()]
    tokenizer=Tokenizer.from_file(str(ROOT/'models/conversation-reference-c7/tokenizer.json'))
    assert sha(ROOT/'models/conversation-reference-c7/tokenizer.json')=='fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d'
    builds={};traces=[];review=[]
    for i,w0 in enumerate(worlds[:88]):
        w=dict(w0);o=i%2;q=(i//2)%2;u=(i//4)%2
        available=[c for c in COLORS if c not in (w['color'],w['color2'])]
        w['correction_color']=available[(i//8)%len(available)]
        for family,compiler in COMPILERS.items():
            equal=bool(((i//8)%2)^o)
            turns,states,factors=compiler(w,o,q,u,equal) if family=='two_objects' else compiler(w,o,q,u)
            item={'id':f'repair-20261012-world-{i:03}-{family}','originalGroupId':f'repair-20261012-world-{i:03}',
                  'split':'train','family':family,'cell':{'orderBit':o,'queryBit':q,'secondaryBit':u},
                  'factors':factors,'messages':[m for user,gold in turns for m in [{'role':'user','content':user},{'role':'assistant','content':gold}]],'stateTrace':states}
            if family=='two_objects':item['cell']['correctionEndsEqual']=equal
            builds[(i,family)]=(turns,item,w);traces.append(item)
            if i<(16 if family=='two_objects' else 8):review.append(item)
    full=[];train=[];originaltrain=[];changedcounts=Counter();messagechanged=Counter();targetchanged=Counter();untouched=[];newtrain=[]
    for line,r in zip(sourcebytes,old):
        if r['split']!='train':full.append(line);continue
        originaltrain.append(line);family=r['source']['family'];i=int(r['groupId'].rsplit('-',1)[-1]);p=r['source']['prefixPairs']
        if family not in FAMILIES:
            out=line;untouched.append(r['id'])
        else:
            turns,item,w=builds[(i,family)];visible=[m for user,gold in turns[:p] for m in [{'role':'user','content':user},{'role':'assistant','content':gold}]]
            if r['source']['neutralAssistantHistory']:
                for m in visible[:-1]:
                    if m['role']=='assistant':m['content']='Entendido.'
            row=deepcopy(r);row['messages']=visible
            row['source'].update({'derivativeDataset':'unidad-pr1-relational-binding-contrast-v1',
              'originalCorpusSha256':EXPECTED,'originalRecordId':r['id'],'originalRecordLineSha256':h(line.decode()),
              'intervention':'state-map relational order/requested-target contrasts only','contrastCell':item['cell'],
              'stateMapGold':True,'newParaphraseStyleFamily':False,'personalData':False,'externalTeacherApiCalled':False})
            out=(json.dumps(row,ensure_ascii=False)+'\n').encode();changedcounts[family]+=1
            messagechanged[family]+=row['messages']!=r['messages']
            targetchanged[family]+=row['messages'][-1]['content']!=r['messages'][-1]['content']
        full.append(out);train.append(out);newtrain.append(json.loads(out))
    assert len(train)==4224 and len(untouched)==2816 and sum(changedcounts.values())==1408
    BASE.mkdir(exist_ok=True)
    paths={'relational-train.jsonl':b''.join(train),'original-train.jsonl':b''.join(originaltrain),
       'relational-full-original-validation.jsonl':b''.join(full),
       'relational-paired-corpus.jsonl':b''.join(train)+newvalbytes,'original-paired-corpus.jsonl':b''.join(originaltrain)+newvalbytes}
    for name,content in paths.items():(BASE/name).write_bytes(content);load_corpus(BASE/name)
    (BASE/'changed-dialogues-state-traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
    (BASE/'semantic-review-seeds.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    devel60path=SOURCEBASE/'pr1-devel-user-exclusion-hashes.json';guard60=json.loads(devel60path.read_text())
    lingpath=ROOT/'private/pr1-wording-history-diagnosis-v1/hash-only-exclusion.json';assert sha(lingpath)=='0279193256fd4aa0b8d49fb49965dbe404ed345407f5991c5304502563a3a407'
    ling=json.loads(lingpath.read_text())
    guard64={x['normalizedUserSha256'] for x in ling['users']};prefix64={x['sha256'] for x in ling['userOnlyPrefixes']}
    valusers={h(norm(m['content'])) for r in newval for m in r['messages'] if m['role']=='user'}
    trainusers={h(norm(m['content'])) for r in newtrain for m in r['messages'] if m['role']=='user'}
    trainprefixes={h(json.dumps([norm(m['content']) for m in r['messages'] if m['role']=='user'],ensure_ascii=False,separators=(',',':'))) for r in newtrain}
    guards={'known60FullUserNormalized':len(trainusers&set(guard60['individualUserNormalizedSha256'])),
            'diagnosis64FullUserNormalized':len(trainusers&guard64),'diagnosis64UserOnlyPrefixes':len(trainprefixes&prefix64),
            'newVal80FullUserNormalized':len(trainusers&valusers)}
    valforms=set();valprefixforms=set()
    for r in newval:
        w=worlds[int(r['groupId'].rsplit('-',1)[-1])];seq=[]
        for m in r['messages']:
            if m['role']=='user':s=skeleton(m['content'],w);valforms.add(s);seq.append(s)
        valprefixforms.add(tuple(seq))
    trainforms=set();trainprefixforms=set()
    for r in newtrain:
        i=int(r['groupId'].rsplit('-',1)[-1]);w=builds[(i,r['source']['family'])][2] if r['source']['family'] in FAMILIES else worlds[i];seq=[]
        for m in r['messages']:
            if m['role']=='user':s=skeleton(m['content'],w);trainforms.add(s);seq.append(s)
        trainprefixforms.add(tuple(seq))
    guards.update({'newValHeldoutSingleUserDelexForms':len(trainforms&valforms),'newValHeldoutWholeUserPrefixDelexForms':len(trainprefixforms&valprefixforms)})
    assert not any(guards.values()),guards
    contexts=[];targets=[];targetbudget=Counter();cells={f:Counter() for f in FAMILIES};exposure={f:Counter() for f in FAMILIES}
    for item in traces:
        f=item['family'];cells[f][tuple(item['cell'].values())]+=1
        for k,v in item['factors'].items():exposure[f][(k,str(v))]+=1
    for r in newtrain:
        ids,mask=dialogue_tokens(r['messages'],tokenizer);inputs,labels=encode_dialogue(r['messages'],tokenizer,512)
        assert len(inputs)==len(ids)-1 and len(inputs)<=512 and labels[-1]==2
        nt=len(tokenizer.encode(r['messages'][-1]['content']).ids)+1;assert nt<=128
        contexts.append(len(inputs));targets.append(nt);targetbudget[r['source']['family']]+=nt
    assert all(set(c.values())==({5,6} if f=='two_objects' else {11}) for f,c in cells.items())
    manifest={'schemaVersion':1,'status':'built-technical-guards-pass-awaiting-semantic-peer',
      'builtAtUtc':datetime.now(timezone.utc).isoformat(),'seed':'No random draws; world index bit factors over original authored seed20261012',
      'originalSourceCorpusSha256':EXPECTED,'originalAuthoringScriptSha256':sha(SOURCEBASE/'frozen_build_repair_corpus.py'),
      'worldMetadataSha256':sha(SOURCEBASE/'reproduced-world-metadata.json'),'compilerSha256':sha(Path(__file__)),
      'files':{p.name:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [*[BASE/n for n in paths],BASE/'changed-dialogues-state-traces.json',BASE/'semantic-review-seeds.json']},
      'primaryPair':{'original':'original-paired-corpus.jsonl','relational':'relational-paired-corpus.jsonl',
         'trainRecordsPerArm':4224,'sameValidationRecordsPerArm':80,'validationSha256':VALSHA,
         'futureTrainingOnly':{'start':'own C7','seed':61,'updatesMaximum':128,'secondsMaximumPerArm':600,'learningRate':0.0001,'validateCheckpointEvery':64,'auxiliaryLossWeights':[0.2,0.2],'objectiveBpeArchitectureRuntime':'unchanged across both arms'},
         'trainingLaunched':False},
      'changedFamilies':sorted(FAMILIES),'changedNativeTrainRecords':dict(changedcounts),
      'changedMessagesRecordCounts':dict(messagechanged),'changedLastTargetRecordCounts':dict(targetchanged),
      'unmodifiedOtherTrainFamilies':8,'byteIdenticalOtherTrainLines':len(untouched),
      'byteIdenticalOriginalValLines':384,'originalRecordSlotsIdsGroupsSplitsOrderRetained':True,
      'originalCorpusSha256After':sha(sourcepath),'originalCorpusUnchanged':sha(sourcepath)==EXPECTED,
      'trainWorldGroups':88,'sameOldValWorldsAlwaysVal':True,'changedFullDialogues':len(traces),
      'semanticReviewSeeds':len(review),'balancedCells':{f:[{'bits':list(k),'worlds':v} for k,v in sorted(c.items())] for f,c in cells.items()},
      'targetExposure':{f:[{'factor':k,'value':v,'dialogues':n} for (k,v),n in sorted(c.items())] for f,c in exposure.items()},
      'perFamilyLastTargetTokenBudgetIncludingEos':dict(targetbudget),'allTrainLastTargetTokensIncludingEos':sum(targetbudget.values()),
      'maximumNativeInputTokens':max(contexts),'maximumLastAssistantTargetsIncludingEos':max(targets),
      'completeContextsWithoutTruncation':True,'assistantEosSupervised':True,
      'guards':guards,'guardSources':{'known60Sha256':sha(devel60path),'diagnosis64HashOnlySha256':sha(lingpath),'newVal80Sha256':VALSHA},
      'semanticCompositionExclusion':'No new heldout cross-task recipe is authored: selected tasks retain alias/activity, present/future place, person relation/singing/crops, and object color/giver operations only. Formal delex/prefix guards complement peer semantic review.',
      'correctionColorPolicy':'44worlds correct to the other object color and44 correct to a third color from original eight-color palette; corrected object always changes, other color and both givers remain intact. Equality and difference are both present, alongside44queries of corrected entity and44of untouched entity.',
      'preferredAddressPolicy':'Identification query uses already-existing TRAIN frame ¿Cuál es mi nombre?; calling/preference queries always target preferred address, never full name.',
      'limitations':['This tests a bounded binding/order/requested-target data contrast, not wording-only improvement or broad conversation.',
        'Four full prefixes are dependent rows; 88worldgroups and352changed dialogues, not1408independent situations.',
        'State-map compiler and technical checks do not replace peer semantic review or model evaluation.',
        'preferred_address secondary cell bit is intentionally unused: four order/query cells have22worlds each; places/people have8cells×11worlds, objects16cells×5–6worlds.',
        'Data volume and update configuration match; changed token lengths change supervised token exposure, explicitly reported.'],
      'torchImported':'torch' in sys.modules,'modelsLoaded':False,'trainingPerformed':False,'coreEdited':False,'ownerChatOrOldSealedFinalRead':False}
    (BASE/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'records':4224,'changed':dict(changedcounts),'guards':guards,
       'maximumContext':max(contexts),'maximumTargetIncludingEos':max(targets),'pairFiles':{n:sha(BASE/n) for n in ('original-paired-corpus.jsonl','relational-paired-corpus.jsonl')},'reviewSeeds':len(review)},ensure_ascii=False))

if __name__=='__main__':main()
