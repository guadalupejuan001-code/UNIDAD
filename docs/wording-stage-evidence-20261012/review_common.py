"""Serialize human-readable project-AI judgments; never used for model input."""
import hashlib,json
from pathlib import Path
def write_review(root,tag,suite,annotations,baseline_f=None):
 root=Path(root);path=root/f'{tag}.{suite}.actual.json';actual=json.loads(path.read_text())
 assert len(actual['episodes'])==len(annotations)==10
 rows=[];turns=[]
 for episode,decisions in zip(actual['episodes'],annotations):
  assert len(episode['turns'])==len(decisions)==4
  for turn,(grade,critical,reason) in zip(episode['turns'],decisions):
   assert grade in ['C','P','F'] and (not critical or grade=='F')
   rows.append({'id':turn['id'],'grade':grade,'critical':bool(critical),'reason':reason});turns.append(turn)
 counts={g:sum(row['grade']==g for row in rows) for g in ['C','P','F']}
 summary={**counts,'criticalFailureTurns':sum(row['critical'] for row in rows),
          'completeDialogs':sum(all(rows[i+j]['grade']=='C' for j in range(4)) for i in range(0,40,4)),
          'requestedTurns':40,'reviewedTurns':40,'actualOutputs':40,
          'budget32ReachedTurns':sum(t['actual']['stopReason']=='token_budget' for t in turns),
          'truncatedPrompts':sum(t['actual']['discardedPromptTokens']>0 for t in turns)}
 if baseline_f is not None:summary['reductionInFVsC7']=(baseline_f-counts['F'])/baseline_f
 review={'modelId':actual['modelId'],'actualSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'suiteSha256':actual['suiteSha256'],'generation':actual['generation'],
         'rubricFile':'unidad-fix/grading-rubric.json','reviewer':{'kind':'project AI','human':False,'independent':False,'blinded':False},
         'turns':rows,'summary':summary,'productionAccepted':False}
 if suite=='known':review['goal50PercentReached']=counts['F']<=14 and summary['criticalFailureTurns']<=6
 (root/f'{tag}.{suite}.review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 print(tag,suite,summary,flush=True)
 return review
