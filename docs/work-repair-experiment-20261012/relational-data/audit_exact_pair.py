"""Independent read-only byte/order/label-exposure checks for both data arms."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/workspace/unidad-conversation-repair');BASE=ROOT/'private/pr1-relational-data-contrast-v1'
SOURCEBASE=ROOT/'private/pr1-wording-composition-audit-v1'
sys.path.insert(0,str(ROOT/'src'))
from tokenizers import Tokenizer
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens,encode_dialogue
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
FAMILIES={'preferred_address','current_and_future_place','two_people','two_objects'}

def exposure(rows,tok):
    allce=Counter();lastce=Counter();contexts=[]
    for row in rows:
        ids,mask=dialogue_tokens(row['messages'],tok)
        inputs,labels=encode_dialogue(row['messages'],tok,512)
        assert len(inputs)==len(ids)-1 and len(inputs)<=512 and labels[-1]==2
        family=row['source']['family'];allce[family]+=sum(mask[1:])
        lastce[family]+=len(tok.encode(row['messages'][-1]['content']).ids)+1
        contexts.append(len(inputs))
    return {'assistantCeTokensIncludingAllAssistantEos':sum(allce.values()),
      'lastAssistantTokensIncludingEos':sum(lastce.values()),
      'byFamilyAllAssistantCeTokens':dict(allce),'byFamilyLastAssistantTokens':dict(lastce),
      'maxInputTokens':max(contexts)}

def main():
    sourcepath=SOURCEBASE/'reproduced-original/repair-corpus.jsonl'
    assert sha(sourcepath)=='862996c131ea17e3623f55530b2f6d61e1d4f64e37672bc99581c19df17a7186'
    oldlines=sourcepath.read_bytes().splitlines(keepends=True);old=[json.loads(x) for x in oldlines]
    fullpath=BASE/'relational-full-original-validation.jsonl';newlines=fullpath.read_bytes().splitlines(keepends=True)
    assert len(oldlines)==len(newlines)==4608
    unchangedfamilies=Counter();unchangedval=0;selectedchanged=Counter();semanticchanged=Counter();lastchanged=Counter()
    for a,b,aline,bline in zip(old,[json.loads(x) for x in newlines],oldlines,newlines):
        assert (a['id'],a['groupId'],a['split'])==(b['id'],b['groupId'],b['split'])
        if a['split']=='validation':assert aline==bline;unchangedval+=1
        elif a['source']['family'] not in FAMILIES:
            assert aline==bline;unchangedfamilies[a['source']['family']]+=1
        else:
            assert a['source']['family']==b['source']['family']
            assert a['source']['neutralAssistantHistory']==b['source']['neutralAssistantHistory']
            assert hashlib.sha256(aline).hexdigest()==b['source']['originalRecordLineSha256']
            selectedchanged[a['source']['family']]+=aline!=bline
            semanticchanged[a['source']['family']]+=a['messages']!=b['messages']
            lastchanged[a['source']['family']]+=a['messages'][-1]['content']!=b['messages'][-1]['content']
    assert len(unchangedfamilies)==8 and sum(unchangedfamilies.values())==2816 and unchangedval==384
    assert set(selectedchanged.values())=={352}
    oldtrainbytes=b''.join(line for line,row in zip(oldlines,old) if row['split']=='train')
    newtrainbytes=b''.join(line for line in newlines if json.loads(line)['split']=='train')
    assert (BASE/'original-train.jsonl').read_bytes()==oldtrainbytes
    assert (BASE/'relational-train.jsonl').read_bytes()==newtrainbytes
    valpath=SOURCEBASE/'prospective-validation/validation.jsonl';valbytes=valpath.read_bytes()
    assert sha(valpath)=='e305adb7fec7c045d2b3ca2d066297aac9dcc04bc5f580c54dae55ac664ab37c'
    assert (BASE/'original-paired-corpus.jsonl').read_bytes()==oldtrainbytes+valbytes
    assert (BASE/'relational-paired-corpus.jsonl').read_bytes()==newtrainbytes+valbytes
    arms={k:load_corpus(BASE/(k+'-paired-corpus.jsonl')) for k in ('original','relational')}
    for k,rows in arms.items():
        assert Counter(r['split'] for r in rows)=={'train':4224,'validation':80}
        assert len({r['groupId'] for r in rows if r['split']=='train'})==88
        assert len({r['groupId'] for r in rows if r['split']=='validation'})==8
    tok=Tokenizer.from_file(str(ROOT/'models/conversation-reference-c7/tokenizer.json'))
    budgets={k:exposure([r for r in rows if r['split']=='train'],tok) for k,rows in arms.items()}
    result={'schemaVersion':1,'status':'all-independent-byte-order-label-exposure-checks-pass',
      'sourceCorpusSha256':sha(sourcepath),'verifierSha256':sha(Path(__file__)),
      'allOriginal4608SlotsIdsGroupsFoldsOrderRetained':True,
      'untouchedTrainFamiliesByteExact':dict(unchangedfamilies),'untouchedTrainLines':2816,'oldValidationLinesByteExact':384,
      'changedNativeTrainLines':dict(selectedchanged),'changedMessagesRecords':dict(semanticchanged),'changedLastTargetsRecords':dict(lastchanged),
      'primaryArms':{k:{'path':str((BASE/(k+'-paired-corpus.jsonl')).relative_to(ROOT)),
        'sha256':sha(BASE/(k+'-paired-corpus.jsonl')),'trainRecords':4224,'validationRecords':80} for k in arms},
      'sameFrozenVal80ExactBytes':True,'sameTrainRecordIdsOrderGroupIdsFoldAssignments':True,
      'originalSourceLinesRetainedInOriginalArm':True,'nativeLoaderPassesBothArms':True,
      'completeContextNoTruncation':True,'targetExposure':budgets,
      'exposureDeltaRelationalMinusOriginal':{'allAssistantCeTokens':budgets['relational']['assistantCeTokensIncludingAllAssistantEos']-budgets['original']['assistantCeTokensIncludingAllAssistantEos'],
        'lastAssistantTokensIncludingEos':budgets['relational']['lastAssistantTokensIncludingEos']-budgets['original']['lastAssistantTokensIncludingEos']},
      'limitation':'Matching records and updates does not match token budget exactly; report changed exposure, do not hide it.',
      'torchImported':'torch' in sys.modules,'modelsLoaded':False,'trainingPerformed':False,'coreEdited':False}
    (BASE/'exact-pair-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'targetExposure':budgets,'delta':result['exposureDeltaRelationalMinusOriginal']},ensure_ascii=False))

if __name__=='__main__':main()
