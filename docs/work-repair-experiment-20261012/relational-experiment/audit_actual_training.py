"""Reconstruct native sampled supervised labels and verify own-weight endpoints."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from ia_propia.conversation import training

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tensors(p):
    with safe_open(str(p), framework='np') as f:
        return {k: f.get_tensor(k) for k in f.keys()}

def equal(a,b):
    return a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)

def main():
    design=json.loads((ROOT/'design.json').read_text())
    assert all(sha(p)==expected for p,expected in design['frozenFilesSha256'].items())
    c7=Path(design['initialModel']); baseline=tensors(c7/'weights.safetensors')
    report={'schemaVersion':1,'createdAtUtc':datetime.now(timezone.utc).isoformat(),'kind':'native_supervised_dose_and_own_weight_integrity','designSha256':sha(ROOT/'design.json'),'helperSha256':sha(__file__),'arms':{},'projectAiVerification':True,'independentHumanReview':False}
    orders=[]
    for arm in design['execution']['armOrder']:
        stage=ROOT/arm; completion=json.loads((stage/'completion.json').read_text()); status=completion['nativeStatus']; run=stage/'run'; cp=run/'checkpoints'/status['lastCheckpoint']; state=json.loads((cp/'state.json').read_text()); cfg=state['training_config']
        records=training._load_training_corpus(design['arms'][arm]['corpus'],cfg)
        tokenizer=Tokenizer.from_file(str(c7/'tokenizer.json'))
        examples=training._encoded(records,tokenizer,512,cfg['window_policy'],**training._encoding_kwargs(cfg))
        train_records=[r for r in records if r['split']=='train']; assert len(examples['train'])==len(train_records)==4224
        assert state['epoch']==0 and state['step']==128 and state['cursor']==1024 and len(state['order'])==4224
        assert state['assistant_targets']=='last_reply_only'
        assert cfg==design['trainingConfig'] and state['model_config']==design['modelConfig']
        order=state['order'][:state['cursor']]; orders.append(order)
        dose=[]
        for i in order:
            inputs,labels=examples['train'][i]
            target=[v for v in labels if v!=-100]
            assert target and target[-1]==2
            dose.append({'recordId':train_records[i]['id'],'inputTokens':len(inputs),'supervisedTargetsIncludingEOS':len(target)})
        zero=[p for p in (ROOT/'all-observed-checkpoints'/arm).iterdir() if json.loads((p/'state.json').read_text())['step']==0]; assert len(zero)==1
        endpoint=tensors(stage/'endpoint/weights.safetensors'); native=tensors(cp/'weights.safetensors')
        assert equal(tensors(zero[0]/'weights.safetensors'),baseline)
        assert equal(endpoint,native) and endpoint.keys()==baseline.keys()
        changed=[k for k in endpoint if not np.array_equal(endpoint[k],baseline[k])]; assert changed
        for n in ('config.json','tokenizer.json'): assert (stage/'endpoint'/n).read_bytes()==(c7/n).read_bytes()
        manifest=json.loads((stage/'endpoint/manifest.json').read_text()); assert manifest['training']['trainedFrom']=='own_checkpoint' and manifest['training']['lineage']['parentModelId']==json.loads((c7/'manifest.json').read_text())['modelId']
        full_epoch=sum(sum(v!=-100 for v in labels) for _,labels in examples['train'])
        report['arms'][arm]={'modelId':manifest['modelId'],'manifestSha256':sha(stage/'endpoint/manifest.json'),'endpointWeightsSha256':sha(stage/'endpoint/weights.safetensors'),'codeCommit':design['codeCommit'],'step':state['step'],'cursor':state['cursor'],'elapsedSeconds':status['elapsedSeconds'],'sourceCheckpoint':cp.name,'nativeInitialAllTensorsExactlyC7':True,'endpointAllTensorsExactlyLastNativeCheckpoint':True,'configTokenizerBytesExactlyC7':True,'trainedTensorCount':len(changed),'tensorCount':len(endpoint),'changedTensorNames':changed,'fullEpochLastReplySupervisedTokensIncludingEOS':full_epoch,'sampledRecords':len(order),'sampledSupervisedTargetsIncludingEOS':sum(d['supervisedTargetsIncludingEOS'] for d in dose),'sampledInputTokens':sum(d['inputTokens'] for d in dose),'samplingOrderIndices':order,'sampledRecordDose':dose,'noRepeatedRecordInFirstEpoch':len(set(order))==len(order),'objective':cfg['objective'],'precisionRuntime':state['train_precision_runtime'],'validationScores':[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines() if json.loads(l)['event']=='validation']}
    report['sameActualRecordIndexOrderAcrossArms']=orders[0]==orders[1]; assert report['sameActualRecordIndexOrderAcrossArms']
    report['frozenInputsUnchanged']=all(sha(p)==expected for p,expected in design['frozenFilesSha256'].items()); assert report['frozenInputsUnchanged']
    output=ROOT/'actual-training-integrity-and-dose.json'; assert not output.exists(); output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'outputSha256':sha(output),'arms':{a:{k:v for k,v in r.items() if k in ('step','elapsedSeconds','sampledRecords','sampledSupervisedTargetsIncludingEOS','sampledInputTokens','fullEpochLastReplySupervisedTokensIncludingEOS','trainedTensorCount')} for a,r in report['arms'].items()},'sameActualOrder':True}))
if __name__=='__main__': main()
