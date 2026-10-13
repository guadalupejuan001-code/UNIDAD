"""Validate explicit human/AI semantic judgments against immutable actual output.

Does not infer quality from token matches, length, loss, or expected answers.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def assemble(actual_path, judgments_path, output_path):
    raw = Path(actual_path).read_bytes()
    actual = json.loads(raw)
    review = json.loads(Path(judgments_path).read_text())
    if review['actualSha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Judgments do not identify these exact generation bytes')
    turns = {t['id']:t for e in actual['episodes'] for t in e['turns']}
    judgments = {j['id']:j for j in review['turns']}
    if len(judgments) != len(review['turns']) or set(turns) != set(judgments):
        raise ValueError('Each actually generated turn needs exactly one judgment')
    counts=Counter()
    for key,j in judgments.items():
        if j['grade'] not in ('C','P','F') or type(j['critical']) is not bool or not j['reason'].strip():
            raise ValueError('Invalid explicit semantic judgment')
        if j['critical'] and j['grade']!='F': raise ValueError('Critical failures require F')
        if not turns[key]['actual']['text'].strip() and j['grade']!='F':
            raise ValueError('Empty output cannot pass')
        counts[j['grade']]+=1
    complete=sum(all(judgments[t['id']]['grade']=='C' for t in e['turns']) and not e.get('invalidEmptyReply') for e in actual['episodes'])
    result={**review,'modelId':actual['modelId'],'suiteSha256':actual['suiteSha256'],
            'generation':actual['generation'],'actualHistoryPreserved':True,
            'summary':{'C':counts['C'],'P':counts['P'],'F':counts['F'],'total':len(turns),
                       'nonComplete':counts['P']+counts['F'],'criticalFailureTurns':sum(j['critical'] for j in judgments.values()),
                       'completeDialogs':complete,'dialogs':len(actual['episodes'])},
            'productionAccepted':False,'testUsedForTraining':False}
    Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'modelId':result['modelId'],'summary':result['summary']},ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--actual',required=True);p.add_argument('--judgments',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();assemble(a.actual,a.judgments,a.output)
