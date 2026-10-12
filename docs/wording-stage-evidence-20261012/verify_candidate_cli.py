"""Exercise the real interactive CLI with frozen inputs and the selected model."""
import contextlib,hashlib,io,json,sys
from pathlib import Path
import torch
from ia_propia.conversation.__main__ import main
root=Path(__file__).resolve().parent
choice=json.loads((root/'candidate-selection.json').read_text())
actual=json.loads((root/f"{choice['label']}.known.actual.json").read_text())
episode=actual['episodes'][6]
inputs='\n'.join(t['user'] for t in episode['turns'])+'\n/salir\n'
old_stdin=sys.stdin
stdout=io.StringIO();stderr=io.StringIO()
torch.set_num_threads(1)
try:
 sys.stdin=io.StringIO(inputs)
 with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
  code=main(['converse','--model',str(root/choice['label']),'--temperature','0','--seed','42','--max-new-tokens','32'])
finally:sys.stdin=old_stdin
transcript=stdout.getvalue()
outputs=[line.split('Unidad: ',1)[1] for line in transcript.splitlines() if 'Unidad: ' in line]
expected=[t['actual']['text'] for t in episode['turns']]
assert code==0 and not stderr.getvalue(),stderr.getvalue()
assert choice['modelId'] in transcript
assert outputs==expected,(outputs,expected)
(root/'candidate-cli-transcript.txt').write_text(transcript)
result={'modelId':choice['modelId'],'label':choice['label'],'exitCode':code,'turns':len(outputs),
        'sameRepliesAsKnownEvaluation':outputs==expected,'cli':'native conversation converse',
        'temperature':0,'seed':42,'maxNewTokens':32,'torchThreads':1,
        'transcriptSha256':hashlib.sha256(transcript.encode()).hexdigest(),'simulatedModel':False}
(root/'candidate-cli-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
