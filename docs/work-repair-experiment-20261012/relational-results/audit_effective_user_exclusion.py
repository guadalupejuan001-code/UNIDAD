"""Post-generation audit of frozen effective DEV80 USERs; no source mutation."""
import hashlib,json,unicodedata,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
norm=lambda s:' '.join(unicodedata.normalize('NFC',s).casefold().split())
h=lambda s:hashlib.sha256(norm(s).encode()).hexdigest()
FIELDS=['name','other','alias','new_alias','city','dest','changed','day','day2','hour','place','first','second','color','color2','title','title2']
def sk(s,w):
    for value,label in sorted([(w[k],'<'+k.upper()+'>') for k in FIELDS],key=lambda x:-len(x[0])):s=s.replace(value,label)
    return norm(s)
def main():
    folder=ROOT/'evaluation'; cases=json.loads((folder/'heldout-val80-cases.json').read_text()); exclusions=json.loads((folder/'heldout-val80-user-exclusion.json').read_text()); freeze=json.loads((folder/'heldout-val80-freeze.json').read_text());
    assert all(sha(folder/k)==v for k,v in freeze['frozenFilesSha256'].items())
    worlds=json.loads((REPO/'private/pr1-wording-composition-audit-v1/reproduced-world-metadata.json').read_text());
    effective={h(t['user']) for e in cases['episodes'] for t in e['turns']}; assert effective=={x['normalizedUserSha256'] for x in exclusions['users']}
    effectiveForms=set(); effectivePrefixes=set(); rawPrefixes=set()
    for e in cases['episodes']:
        w=worlds[int(e['sourceGroupId'].rsplit('-',1)[1])]; users=[]
        for t in e['turns']:
            users.append(t['user']); effectiveForms.add(sk(t['user'],w)); effectivePrefixes.add(tuple(sk(u,w) for u in users)); rawPrefixes.add(tuple(norm(u) for u in users))
    report={'schemaVersion':1,'createdAtUtc':datetime.now(timezone.utc).isoformat(),'scope':'post-generation exclusion audit; not pre-training protocol or FINAL','casesSha256':sha(folder/'heldout-val80-cases.json'),'freezeSha256':sha(folder/'heldout-val80-freeze.json'),'helperSha256':sha(__file__),'effectiveUsers':80,'arms':{},'limits':['Exact and field-delexicalized whole forms only; does not prove semantic family independence.','Original eightVAL worlds reused as prescribed; no world moved toTRAIN.','Additional third-color replacement in two_objects derivative is not delexicalized by this supplemental audit. Both changed effectiveUSERs are in other families; original DATA author guards cover native corpus before training.','All80 effectiveUSERs/gold remain development and were validation inputs for loss; native optimizer trained on TRAIN only.']}
    for arm in ('original','relational'):
        path=REPO/'private/pr1-relational-data-contrast-v1'/f'{arm}-paired-corpus.jsonl'; rows=[json.loads(l) for l in path.read_text().splitlines()]; train=[r for r in rows if r['split']=='train']; trainHash=set(); trainForms=set(); trainPrefix=set(); trainRawPrefixes=set()
        for r in train:
            w=worlds[int(r['groupId'].rsplit('-',1)[1])]; users=[m['content'] for m in r['messages'] if m['role']=='user']; trainHash.update(h(u) for u in users); trainForms.update(sk(u,w) for u in users); trainPrefix.add(tuple(sk(u,w) for u in users)); trainRawPrefixes.add(tuple(norm(u) for u in users))
        overlaps={'effectiveNormalizedUserHashes':len(effective&trainHash),'wholeUserDelexicalizedForms':len(effectiveForms&trainForms),'wholeUserPrefixDelexicalizedForms':len(effectivePrefixes&trainPrefix),'normalizedUserOnlyPrefixes':len(rawPrefixes&trainRawPrefixes)}
        report['arms'][arm]={'corpusSha256':sha(path),'trainRecords':len(train),'overlapCounts':overlaps,'changedUserChecks':[{**x,'effectiveUSERexactInTRAIN':x['effectiveNormalizedUserSha256'] in trainHash} for x in exclusions['changedUsers']]}
        assert not any(overlaps.values()),overlaps
    assert all(sha(folder/k)==v for k,v in freeze['frozenFilesSha256'].items()); report['frozenEvaluationFilesUnchanged']=True; p=folder/'effective-user-exclusion-audit.json'; assert not p.exists(); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'sha256':sha(p),'arms':report['arms']}))
if __name__=='__main__':main()
