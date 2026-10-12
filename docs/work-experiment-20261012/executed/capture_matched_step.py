"""Retain the prospectively amended common step, without reading generation scores."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from ia_propia.conversation.artifacts import save_artifact
from ia_propia.conversation.training import _load_checkpoint,_origin_metadata,_loss_metadata

root=Path(__file__).resolve().parent
torch.set_num_threads(1)
pending={'v1','v2'}
deadline=time.monotonic()+900
while pending and time.monotonic()<deadline:
    for arm in list(pending):
        run=root/f'pointer-{arm}-run'
        for check in sorted((run/'checkpoints').glob('step-000000128-*')):
            try:
                # Only committed checkpoint directories satisfy the native manifest verifier.
                model,tokenizer,state,_=_load_checkpoint(run,check.name)
            except (FileNotFoundError,ValueError):continue
            if state['step']!=128:raise RuntimeError('Wrong comparison step')
            metadata={'corpusSha256':state['corpus_sha256'],'selectedStep':128,
                      'selectionSplit':'prospective_common_step_before_quality_observed',
                      'config':state['training_config'],'assistantTargets':state['assistant_targets'],
                      **_origin_metadata(state),**_loss_metadata(state['training_config'])}
            out=root/f'pointer-{arm}-step128'
            m=save_artifact(out,model,tokenizer,training=metadata)
            # Preserve the exact resumable common checkpoint too.
            preserved=root/f'pointer-{arm}-resumable128'
            shutil.copytree(check,preserved)
            print(json.dumps({'arm':arm,'step':128,'modelId':m['modelId']}),flush=True)
            pending.remove(arm)
            break
    if pending:time.sleep(1)
if pending:raise RuntimeError('Missing step128 snapshots: '+repr(pending))
