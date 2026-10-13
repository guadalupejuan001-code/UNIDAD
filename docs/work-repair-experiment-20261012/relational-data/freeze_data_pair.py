"""Freeze already-reviewed data bytes; no ML runner, models, or training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/unidad-conversation-repair');BASE=ROOT/'private/pr1-relational-data-contrast-v1'
SOURCEBASE=ROOT/'private/pr1-wording-composition-audit-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def main():
    build=read(BASE/'build-manifest.json');audit=read(BASE/'exact-pair-audit.json')
    relations=read(BASE/'peer-review-relations.json');objects=read(BASE/'peer-review-objects.json');addendum=read(BASE/'peer-review-objects-addendum.json')
    assert relations['all24ReadInFull'] and relations['compilerPathsReviewed'] and relations['all96GoldAnswersCheckedAgainstStateMaps']
    assert relations['sourceCompilerSha256']==sha(BASE/'build_relational_contrast.py')
    assert relations['sourceSeedsSha256']==sha(BASE/'semantic-review-seeds.json')
    assert relations['sourceTrainSha256']==sha(BASE/'relational-train.jsonl')
    assert objects['all16ReadInFull'] and objects['compilerPathsReviewed']
    assert addendum['priorReceiptSHA']==sha(BASE/'peer-review-objects.json')
    assert addendum['unchanged16Objects'] and addendum['objectsCompilerBranchUnchanged']
    for name,digest in addendum['currentSourceHashes'].items():assert sha(BASE/name)==digest
    for name,digest in objects['sourceHashes'].items():
        p=BASE/name if name=='relational-train.jsonl' else BASE/'peer-reviewed-pre-unknown-fix'/name
        assert sha(p)==digest
    decisions=relations['records']+objects['verdicts'];seeds=read(BASE/'semantic-review-seeds.json')
    assert len(decisions)==40 and {r['id'] for r in decisions}=={r['id'] for r in seeds}
    assert all(r['verdict']=='accept' for r in decisions)
    before=read(BASE/'peer-reviewed-pre-unknown-fix/semantic-review-seeds.json')
    assert [r['messages'] for r in before]==[r['messages'] for r in seeds]
    assert [r for r in before if r['family']=='two_objects']==[r for r in seeds if r['family']=='two_objects']
    assert not any(build['guards'].values())
    for item in build['files'].values():assert sha(ROOT/item['path'])==item['sha256']
    assert build['compilerSha256']==sha(BASE/'build_relational_contrast.py')
    for arm,item in audit['primaryArms'].items():assert sha(ROOT/item['path'])==item['sha256']
    assert audit['sameFrozenVal80ExactBytes'] and audit['allOriginal4608SlotsIdsGroupsFoldsOrderRetained']
    assert build['maximumNativeInputTokens']<=512 and build['maximumLastAssistantTargetsIncludingEos']<=128
    original=SOURCEBASE/'reproduced-original/repair-corpus.jsonl'
    assert sha(original)=='862996c131ea17e3623f55530b2f6d61e1d4f64e37672bc99581c19df17a7186'
    newval=SOURCEBASE/'prospective-validation/validation.jsonl'
    assert sha(newval)=='e305adb7fec7c045d2b3ca2d066297aac9dcc04bc5f580c54dae55ac664ab37c'
    outputpaths=[*[ROOT/item['path'] for item in build['files'].values()],
      BASE/'build_relational_contrast.py',BASE/'audit_exact_pair.py',Path(__file__),
      BASE/'build-manifest.json',BASE/'exact-pair-audit.json',BASE/'peer-review-relations.json',
      BASE/'peer-review-objects.json',BASE/'peer-review-objects-addendum.json']
    manifest={'schemaVersion':1,'status':'frozen-ready-data-for-parent-controlled-pair-no-training',
      'frozenAtUtc':datetime.now(timezone.utc).isoformat(),
      'primaryPair':build['primaryPair'],
      'primaryArms':audit['primaryArms'],
      'files':{str(p.relative_to(BASE)):{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in outputpaths},
      'originalSource':{'path':str(original.relative_to(ROOT)),'sha256':sha(original),'archivedAllOriginalRecords':4608,
        'unchanged':True,'recordsMoved':0,'originalAuthoringScriptSha256':build['originalAuthoringScriptSha256'],
        'worldMetadataSha256':build['worldMetadataSha256']},
      'validationSource':{'path':str(newval.relative_to(ROOT)),'sha256':sha(newval),'records':80,
        'developmentOnly':True,'bothArmsExactSameBytes':True},
      'trainRecordsPerArm':4224,'trainWorldGroupsPerArm':88,'changedFamilies':build['changedFamilies'],
      'changedNativeTrainLines':audit['changedNativeTrainLines'],
      'changedMessagesRecords':audit['changedMessagesRecords'],'changedLastTargetRecords':audit['changedLastTargetsRecords'],
      'byteExactUntouchedTrainLines':2816,'byteExactUntouchedTrainFamilies':8,'byteExactOldValidationLines':384,
      'allSlotsIdsOrderGroupIdsAndSplitsRetained':True,
      'bounds':{'maximumTrainInputTokens':253,'maximumLastTargetTokensIncludingEos':27,
        'maximumSharedValidationInputTokens':260,'completeNoTruncation':True,'nativeLoaderPassesBothArms':True,
        'existingC7BpeSha256':'fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d'},
      'balancedCells':build['balancedCells'],'targetExposureFactors':build['targetExposure'],
      'assistantCeExposure':audit['targetExposure'],'exposureDelta':audit['exposureDeltaRelationalMinusOriginal'],
      'guards':build['guards'],'guardSources':build['guardSources'],
      'semanticCompositionExclusion':build['semanticCompositionExclusion'],
      'review':{'projectAiOnly':True,'independentHumanReview':False,'acceptedSemanticSeeds':40,'rejectedSemanticSeeds':0,
        'goldResponsesReadAndChecked':160,'compilerBranchesReviewed':True,'changedRealizedDialogues':352,
        'scope':'40 representative factor seeds and compiler paths reviewed; not an exhaustive manual reread of all352 realized dialogues.',
        'unknownStateCorrection':'Unobserved singing is None, never unsupported False. No USER, gold or native corpus bytes changed.',
        'preservedPriorReviewSnapshots':{'directory':'peer-reviewed-pre-unknown-fix',
          'compilerSha256':sha(BASE/'peer-reviewed-pre-unknown-fix/build_relational_contrast.py'),
          'seedsSha256':sha(BASE/'peer-reviewed-pre-unknown-fix/semantic-review-seeds.json'),
          'stateTracesSha256':sha(BASE/'peer-reviewed-pre-unknown-fix/changed-dialogues-state-traces.json')}},
      'provenance':{'kind':'derivative of first-party authored-synthetic fictional TRAIN worlds',
        'stateMapsForGold':True,'newParaphraseStyleFamilies':False,'externalTeacherApiCalls':False,
        'ownerChatsUsed':False,'diagnosticOrHeldoutGoldUsedForAuthoring':False,
        'developmentUserHashesAndHeldoutUserSkeletonsUsedForExclusionOnly':True},
      'limits':build['limitations']+[
        'two_objects thirdQueriesCorrectedEntity and lastQueriesKnownGiver both equal q==u; their individual effects are collinear.',
        'Only one object has a disclosed giver in each dialogue; missing/known-giver results are balanced but not an exhaustive gift-state factorial.',
        'All four families change as one DATA block; this trial cannot identify each sub-change as a separate causal effect.',
        'The eight unmodified families inherit archived source ambiguities, including project ownership and daily sequencing; no silent correction is made.',
        'All40 reviewed seeds cover semantic cells and all three original style branches, but not the exhaustive style-by-cell cross product.',
        'No improvement of generation, linguistic foundation or final conversation acceptance has yet been measured.'],
      'coreOrHeadOrRuntimeChanges':False,'modelsLoaded':False,'trainingLaunched':False,'mlRunnerCreated':False,
      'ownerChatOrOldSealedFinalRead':False}
    out=BASE/'freeze-manifest.json';out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'freezeManifestSha256':sha(out),
      'primaryPairHashes':{k:v['sha256'] for k,v in manifest['primaryArms'].items()},'acceptedSeeds':40,
      'compilerSha256':sha(BASE/'build_relational_contrast.py')},ensure_ascii=False))

if __name__=='__main__':main()
