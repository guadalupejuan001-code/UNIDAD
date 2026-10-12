"""Audit actual reproduced PR1 rows, separating world/wording/event/binding overlap."""
import json,pathlib,hashlib,collections,re,unicodedata,sys,time
ROOT=pathlib.Path('/workspace/unidad-conversation-repair');BASE=ROOT/'private/pr1-wording-composition-audit-v1';CORPUS=BASE/'reproduced-original/repair-corpus.jsonl';worlds=json.loads((BASE/'reproduced-world-metadata.json').read_text());rows=[json.loads(x) for x in CORPUS.read_bytes().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if sha(CORPUS)!='862996c131ea17e3623f55530b2f6d61e1d4f64e37672bc99581c19df17a7186':raise ValueError('Source corpus identity differs')
FIELDS=['name','other','alias','new_alias','city','dest','changed','day','day2','hour','place','first','second','color','color2','title','title2']
RECIPES={
'preferred_address':['set_preferred_alias','distracting_desired_activity','replace_preferred_alias','query_prior_and_current_alias'],
'missing_information':['unknown_color','reveal_color','reveal_giver','query_known_color_and_unknown_price'],
'current_and_future_place':['current_and_future_place','replace_future_place','query_future_as_current_with_current_restated','query_current_and_future_place'],
'listening':['listening_without_advice','grievance_and_question_request','reveal_colleague_not_sibling','query_grievance_without_advice'],
'project_change':['project_name_and_drawings','grandmother_origin','rename_preserve_drawings','query_grandmother_relation'],
'two_people':['sibling_and_neighbor_mapping','sibling_sings_neighbor_grows_roses','correct_sibling_roses_neighbor_tulips','query_sibling_identity_and_neighbor_activity'],
'reschedule':['day_and_hour','replace_day_preserve_hour','reveal_place','query_current_day_hour_place'],
'task_order':['first_second_order','swap_order','query_later_activity_no_execution','query_prior_and_current_first_activity'],
'unknown_author':['book_title','unknown_author','class_purpose','query_title_purpose_and_unknown_author'],
'daily_sequence':['morning_first_activity_relaxes','second_after_first','move_first_activity_to_evening','query_order_relation'],
'two_objects':['bag_color_and_watch_color','reveal_bag_giver','correct_watch_to_bag_color','query_unknown_watch_giver'],
'limits_and_identity':['assistant_name','user_name','unknown_user_address','query_disclosed_user_name']}
BINDINGS={
'preferred_address':['name','alias','new_alias','first'],'missing_information':['name','color'],'current_and_future_place':['city','dest','changed'],
'listening':['name'],'project_change':['title','title2'],'two_people':['name','other'],'reschedule':['name','day','day2','hour','place'],
' task_order':['first','second'],'task_order':['first','second'],'unknown_author':['title'],'daily_sequence':['first','second'],'two_objects':['name','color','color2'],'limits_and_identity':['name']}
def skeleton(messages,w):
 values=sorted(((w[k],'<'+k.upper()+'>') for k in FIELDS),key=lambda x:-len(x[0]));out=[]
 for m in messages:
  text=m['content']
  for value,label in values:text=text.replace(value,label)
  out.append({'role':m['role'],'content':text})
 return out
sig={fold:{name:set() for name in ['world','family','style','single_user_form','prefix_user_form','full_dialogue_form','event_recipe','full_world_binding']} for fold in ['train','validation']};perfamily={};neutral=collections.Counter();actual_replacement=collections.Counter();hist_replaced_positions=collections.Counter();bindings={f:{fold:set() for fold in ['train','validation']} for f in RECIPES}
for r in rows:
 fold=r['split'];i=int(r['groupId'].rsplit('-',1)[-1]);w=worlds[i];fam=r['source']['family'];p=r['source']['prefixPairs'];sm=skeleton(r['messages'],w);users=[m['content'] for m in sm if m['role']=='user'];single=users[-1]
 sig[fold]['world'].add(r['groupId']);sig[fold]['family'].add(fam);sig[fold]['style'].add(w['style']);sig[fold]['single_user_form'].add((fam,p,single));sig[fold]['prefix_user_form'].add((fam,tuple(users)));sig[fold]['full_dialogue_form'].add((fam,json.dumps(sm,ensure_ascii=False,sort_keys=True)));sig[fold]['event_recipe'].add(tuple(RECIPES[fam]));sig[fold]['full_world_binding'].add(tuple(w[k] for k in FIELDS))
 bindings[fam][fold].add(tuple(w[k] for k in BINDINGS[fam]))
 pf=perfamily.setdefault(fam,{s:{'worlds':set(),'styles':collections.Counter(),'single_user_forms':{i:set() for i in range(1,5)},'prefix_user_forms':set(),'full_dialogue_forms':set(),'event_recipes':set(),'rows':0} for s in ['train','validation']})[fold]
 pf['rows']+=1;pf['worlds'].add(r['groupId']);pf['styles'][w['style']]+=1;pf['single_user_forms'][p].add(single);pf['prefix_user_forms'].add(tuple(users));pf['full_dialogue_forms'].add(json.dumps(sm,ensure_ascii=False,sort_keys=True));pf['event_recipes'].add(tuple(RECIPES[fam]))
 if r['source']['neutralAssistantHistory']:
  neutral[fold]+=1;positions=sum(m['role']=='assistant' and m['content']=='Entendido.' for m in r['messages'][:-1]);hist_replaced_positions[fold]+=positions
  if positions:actual_replacement[fold]+=1
for fam,v in perfamily.items():
 for fold,x in v.items():
  x['worlds']=len(x['worlds']);x['styles']=dict(x['styles']);x['single_user_form_count_by_prefix']={p:len(s) for p,s in x['single_user_forms'].items()};x['prefix_user_form_count']=len(x['prefix_user_forms']);x['full_dialogue_form_count']=len(x['full_dialogue_forms']);x['event_recipe_count']=len(x['event_recipes']);x['single_user_forms']={p:sorted(s) for p,s in x['single_user_forms'].items()};x.pop('prefix_user_forms');x.pop('full_dialogue_forms');x.pop('event_recipes')
 perfamily[fam]['factual_binding_overlap']={'fields':BINDINGS[fam],'trainDistinct':len(bindings[fam]['train']),'validationDistinct':len(bindings[fam]['validation']),'validationBindingsSeenTrain':len(bindings[fam]['train']&bindings[fam]['validation']),'validationBindingsNotSeenTrain':len(bindings[fam]['validation']-bindings[fam]['train'])}
overlap={name:{'trainDistinct':len(sig['train'][name]),'validationDistinct':len(sig['validation'][name]),'validationSeenInTrain':len(sig['validation'][name]&sig['train'][name]),'validationUnseenInTrain':len(sig['validation'][name]-sig['train'][name])} for name in sig['train']}
# Guard stores USER hashes only from already-exposed PR DEVEL40+DEVEL20.
guardraw=set();guardnorm=set();input_guards=[]
for name in ['known-dev.json','fresh-probe.json']:
 p=ROOT/'docs/work-experiment-20261012'/name;d=json.loads(p.read_text());users=[t['user'] for e in d['episodes'] for t in e['turns']]
 guardraw.update(hashlib.sha256(s.encode()).hexdigest() for s in users);guardnorm.update(hashlib.sha256(' '.join(unicodedata.normalize('NFC',s).casefold().split()).encode()).hexdigest() for s in users);input_guards.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'userTurnsHashed':len(users),'use':'USERstrings for lexical exclusion only; no reference replies/rubrics used for data authoring'})
 # Source labels/reviews are not inspected or included.
guard={'schemaVersion':1,'scope':'Known40andfresh20alreadyexposedDEVELOPMENT, not sealedFINAL','inputs':input_guards,'individualUserRawSha256':sorted(guardraw),'individualUserNormalizedSha256':sorted(guardnorm),'ownerChatOrOldSealedFinalRead':False,'generatedOutputsOrConstraintsUsed':False}
(BASE/'pr1-devel-user-exclusion-hashes.json').write_text(json.dumps(guard,ensure_ascii=False,indent=2)+'\n')
result={'schemaVersion':1,'kind':'actual_pr1_wording_event_composition_and_world_overlap_audit','createdAtUtc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prHead':'eb0f811aabdf4ba5bfe8957ceeaeebc942b9adb5','sourceCorpusSha256':sha(CORPUS),'originalAuthoringScriptSha256':sha(BASE/'frozen_build_repair_corpus.py'),'reproductionReceiptSha256':sha(BASE/'reproduction-receipt.json'),'records':dict(collections.Counter(r['split'] for r in rows)),'representedOverlap':overlap,'perFamily':perfamily,'eventRecipes':RECIPES,'neutralAssistantHistory':{s:{'records':sum(r['split']==s for r in rows),'flagTrueRows':neutral[s],'actuallyReplacedEarlierAssistantRows':actual_replacement[s],'replacedEarlierAssistantPositions':hist_replaced_positions[s],'flagFraction':neutral[s]/sum(r['split']==s for r in rows),'actualReplacementFraction':actual_replacement[s]/sum(r['split']==s for r in rows)} for s in ['train','validation']},'lexicalGuardReceipt':{'path':str((BASE/'pr1-devel-user-exclusion-hashes.json').relative_to(ROOT)),'sha256':sha(BASE/'pr1-devel-user-exclusion-hashes.json')},'definitions':{'world':'Entire original worldgroup; oldsplit88/8protects this identity only.','single_user_form':'Perfamily/prefix sentence after replacing exact sampled factual values by role-specific placeholders.','prefix_user_form':'Whole USERprefix sequence after samevalue replacement.','full_dialogue_form':'All USER+ASSISTANTtext, neutral acknowledgements included; targets are present for representation audit, not modelgrading.','event_recipe':'Declared semantic operation/query sequence of the12authoredtasks; value-independent, distinct from world count.','factual_binding':'Perfamily relevant ordered tuple of sampled facts. Partly new bindings are not unseen wording or eventcomposition.'},'limitations':['Exact delexicalized equality is not semantic paraphrase analysis.','English/Spanish fluency or conversation acceptance is not measured by worldcount/datareproduction.','Old16TRAIN64stepcontrols and prior8paraphraseprobe are not reexecuted.','Validationworlds are disjoint but taskfamilies, formulation skeletons and eventrecipes are shared.','Entendido earlier-history replacement supplies no false assistantclaims and does not establish resistance to contradictory actualgeneratedhistory.','Known40+fresh20 are exposedDEVEL; no FINALacceptanceclaim is made.'],'trainingPerformed':False,'modelOrCoreChanges':False,'torchImported':'torch' in sys.modules,'oldMainPrivateTouched':False,'oldSealedFinalOrOwnerCorpusRead':False}
(BASE/'represented-split-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'overlap':overlap,'neutral':result['neutralAssistantHistory'],'perFamilyBindings':{f:v['factual_binding_overlap'] for f,v in perfamily.items()},'userGuardHashCounts':{'raw':len(guardraw),'norm':len(guardnorm)}},ensure_ascii=False))
