"""ABBA timing of identical next-position distributions, same CPU and weights."""
import json
import statistics
import time
from pathlib import Path
import torch
from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import encode_prompt

root=Path(__file__).resolve().parent
torch.set_num_threads(1)
engine=ConversationEngine(root.parent/'UNIDAD/models/conversation-reference-c7')
evidence=json.loads((root/'C7-selected.actual.json').read_text())
prompts=[evidence['episodes'][0]['turns'][0]['inputMessages'],evidence['episodes'][5]['turns'][3]['inputMessages']]
result={'modelId':engine.model_id,'torchThreads':1,'repeatsPerBatch':12,'order':['full','last','last','full'],
        'fullGenerationsByteIdentical':evidence['baselineExactMatches'],'generationsCompared':40,'prompts':[]}
model=engine._model; model.eval()
with torch.inference_mode():
 for messages in prompts:
    ids,_=encode_prompt(messages,engine._tokenizer,model.config['context_length'])
    inputs=torch.tensor([ids]);mask=torch.zeros_like(inputs,dtype=torch.bool);mask[:,-1]=True
    def full():return model(inputs)[0,-1]
    def last():return model.selected_logits(inputs,mask)[0]
    reference=full(); optimized=last()
    torch.testing.assert_close(reference,optimized,rtol=1e-5,atol=1e-5)
    timings=[]
    for mode in result['order']:
        fn=full if mode=='full' else last
        fn();start=time.perf_counter();cpu=time.process_time()
        for _ in range(12):fn()
        timings.append({'mode':mode,'wallSeconds':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu})
    seconds={mode:statistics.mean(t['wallSeconds'] for t in timings if t['mode']==mode) for mode in ['full','last']}
    result['prompts'].append({'tokens':len(ids),'timings':timings,'meanWallSeconds':seconds,
                              'speedup':seconds['full']/seconds['last'],
                              'maxLogitAbsDifference':float((reference-optimized).abs().max())})
(root/'projection-benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
