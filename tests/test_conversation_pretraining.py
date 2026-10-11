"""Meaningful small-fixture objective tests; no external corpora/model weights."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training

HAS_EXTRAS = all(importlib.util.find_spec(p) is not None for p in ("torch", "tokenizers", "safetensors"))


def document(i, split, text=None):
    return {"id": f"doc-{i}", "groupId": f"article-{i}", "split": split,
            "source": {"dataset":"authored-text-test", "kind":"authored-synthetic", "note":"METADATA_NEVER_TARGET"},
            "text": text or f"Este es el documento ficticio {i}. Explica palabras distintas con calma y precisión. "*3}


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class TextEncodingTests(unittest.TestCase):
    def test_train_only_bpe_without_metadata_or_system_and_byte_roundtrip(self):
        from ia_propia.conversation.tokenizer import fit_text_tokenizer, text_tokens
        from ia_propia.conversation.contract import DEFAULT_SYSTEM_PROMPT
        rows = [document(i,"train","El café de mañana es tranquilo. "*20) for i in range(3)]
        tokenizer = fit_text_tokenizer(rows, 400)
        raw = json.loads(tokenizer.to_str())
        joined = " ".join(raw["model"]["vocab"])
        self.assertNotIn("METADATA_NEVER_TARGET", joined)
        tokens = text_tokens("  Árbol\nPingüino y café. ", tokenizer)
        self.assertEqual(tokens[0], 1); self.assertEqual(tokens[-1], 2)
        self.assertNotIn(0, tokens); self.assertNotIn(3, tokens); self.assertNotIn(4, tokens); self.assertNotIn(5, tokens)
        self.assertEqual(tokenizer.decode(tokens[1:-1]), "  Árbol\nPingüino y café. ")
        with self.assertRaises(ValueError): fit_text_tokenizer([document(99,"validation")], 400)
        with self.assertRaises(ValueError): fit_text_tokenizer([], 400)
        # Serialization is deterministic if validation text is never supplied.
        same = fit_text_tokenizer(rows, 400)
        self.assertEqual(tokenizer.to_str(), same.to_str())
        self.assertNotIn(DEFAULT_SYSTEM_PROMPT, [r["text"] for r in rows])

    def test_fulltext_shift_before_windows_exact_positions_eos_and_padding(self):
        # Unique payload IDs let this fixture audit GLOBAL target positions,
        # independently of repeated natural-language IDs and tokenizer merges.
        from ia_propia.conversation.tokenizer import fit_text_tokenizer
        rows = [document(i, split) for i,split in enumerate(["train","validation","test"])]
        tokenizer = fit_text_tokenizer(rows[:1], 300)
        for context in [8, 512]:
            for length in [context-1,context,context+1,2*context-1,2*context,2*context+1]:
                tokens = [1,*range(6,6+length-1),2]  # len(input_ids)==length.
                for policy in ["contiguous_nonoverlapping", "sliding_context"]:
                    with self.subTest(context=context,length=length,policy=policy), patch(
                        "ia_propia.conversation.tokenizer.text_tokens", return_value=tokens):
                        encoded = training._encoded(rows,tokenizer,context,policy,objective="causal_lm")
                    for windows in encoded.values():
                        taught = [value for _,labels in windows for value in labels if value!=-100]
                        self.assertEqual(taught,tokens[1:])
                        self.assertEqual(taught.count(2),1)
                        self.assertNotIn(1,taught); self.assertNotIn(0,taught)
                        self.assertTrue(all(len(x)==len(y)<=context for x,y in windows))
                        self.assertTrue(all(x[0] in tokens[:-1] for x,_ in windows))
        import torch
        x,y=training._batch([([1,9,8],[9,8,2]),([1],[2])],[0,1],torch)
        self.assertEqual(x[1].tolist(),[1,0,0]); self.assertEqual(y[1].tolist(),[2,-100,-100])


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class PretrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.rows=[document(i,split) for i,split in enumerate(["train","train","validation","test"])]
        self.corpus=self.root/"text.jsonl"; self.write(self.corpus,self.rows)
        self.model={"vocab_size":300,"d_model":8,"n_layers":1,"n_heads":2,"ffn_multiplier":1,
                    "context_length":8,"dropout":.2,"seed":9}
        self.config={"batch_size":2,"max_steps":5,"max_epochs":50,"validation_every":2,
                     "checkpoint_every":1,"patience":100,"min_delta":0,"seed":7,
                     "warmup_steps":2,"torch_threads":1,"objective":"causal_lm"}

    def tearDown(self): self.temp.cleanup()

    def write(self,path,rows): path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")

    def test_fulltext_resume_exact_weights_optimizer_rng_and_objective_rejection(self):
        import torch
        for objective,policy in [("causal_lm","contiguous_nonoverlapping"),("causal_lm","sliding_context"),
                                 ("assistant_last_reply","sliding_context")]:
            config={**self.config,"objective":objective,"window_policy":policy,"projection_mode":"supervised"}
            path=self.corpus
            if objective=="assistant_last_reply":
                path=self.root/"sft.jsonl"
                self.write(path,[{k:v for k,v in row.items() if k!="text"} | {"messages":[
                    {"role":"user","content":f"Dato ficticio {i}."},{"role":"assistant","content":row["text"]}]}
                    for i,row in enumerate(self.rows)])
            name=objective+policy; continuous_dir=self.root/(name+"continuous"); paused_dir=self.root/(name+"paused")
            continuous=training.train(path,continuous_dir,model_config=self.model,training_config=config)
            original=training._checkpoint; stopped=[]
            def checkpoint(*args,**kwargs):
                result=original(*args,**kwargs)
                if args[5]["step"]==2 and not stopped:
                    stopped.append(True); os.kill(os.getpid(),signal.SIGTERM)
                return result
            with patch.object(training,"_checkpoint",side_effect=checkpoint):
                paused=training.train(path,paused_dir,model_config=self.model,training_config=config)
            self.assertEqual(paused["step"],2)
            resumed=training.train(path,paused_dir,resume=True)  # Saved objective without caller config.
            a,_,sa,pa=training._load_checkpoint(continuous_dir,continuous["lastCheckpoint"])
            b,_,sb,pb=training._load_checkpoint(paused_dir,resumed["lastCheckpoint"])
            for key,value in a.state_dict().items(): self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
            for index,moments in pa["optimizer"]["state"].items():
                for key,value in moments.items(): self.assertTrue(torch.equal(value,pb["optimizer"]["state"][index][key]))
            self.assertEqual(pa["scheduler"],pb["scheduler"]); self.assertTrue(torch.equal(pa["torch_rng"],pb["torch_rng"]))
            for key in ["python_rng","order","cursor","epoch","best_validation_nll"]: self.assertEqual(sa[key],sb[key])
            self.assertEqual(sa["assistant_targets"],"all_text_tokens" if objective=="causal_lm" else "last_reply_only")
            changed={**config,"objective":"assistant_last_reply" if objective=="causal_lm" else "causal_lm"}
            with self.assertRaisesRegex(ValueError,"configuration mismatch"):
                training.train(path,paused_dir,resume=True,training_config=changed)
            examples=training._encoded(self.rows if objective=="causal_lm" else [json.loads(l) for l in path.read_text().splitlines()],
                training._load_checkpoint(paused_dir,resumed["lastCheckpoint"])[1],8,policy,objective=objective)
            broken=copy.deepcopy(sb); broken["assistant_targets"]="last_reply_only" if objective=="causal_lm" else "all_text_tokens"
            with self.assertRaisesRegex(ValueError,"supervision"): training._validate_resume_state(broken,examples)

    def test_training_fits_only_train_exports_correct_objective_and_finalizes_explicitly(self):
        from ia_propia.conversation.tokenizer import fit_text_tokenizer
        from ia_propia.conversation.artifacts import load_artifact
        captured=[]
        def fit(rows,**kwargs):
            rows=list(rows); captured.extend(rows); return fit_text_tokenizer(rows,**kwargs)
        with patch("ia_propia.conversation.tokenizer.fit_text_tokenizer",side_effect=fit):
            result=training.train(self.corpus,self.root/"run",model_config=self.model,training_config=self.config)
        self.assertEqual(captured,[r for r in self.rows if r["split"]=="train"])
        exported=training.export_best(self.root/"run",self.root/"snapshot")
        self.assertEqual(exported["training"]["trainingObjective"],"causal_lm")
        self.assertEqual(exported["training"]["assistantTargets"],"all_text_tokens")
        self.assertFalse(exported["training"]["testEvaluated"])
        self.assertFalse((self.root/"run/evaluation.json").exists())
        _,_,loaded=load_artifact(self.root/"snapshot")
        self.assertEqual(loaded["modelId"],exported["modelId"])
        evaluated=training.finalize(self.root/"run",export_dir=self.root/"final")
        self.assertEqual(evaluated["assistantTargets"],"all_text_tokens")
        self.assertEqual(evaluated["trainingObjective"],"causal_lm")
        self.assertFalse(evaluated["testUsedForSelection"])
        final=load_artifact(self.root/"final")[2]
        self.assertEqual(final["training"]["assistantTargets"],"all_text_tokens")

    def test_new_sft_stage_preserves_own_pretrained_weights_and_tokenizer_without_resize(self):
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        training.train(self.corpus,self.root/"pretrain",model_config=self.model,training_config=self.config)
        training.export_best(self.root/"pretrain",self.root/"parent")
        parent,tokenizer,manifest=load_artifact(self.root/"parent")
        path=self.root/"sft.jsonl"
        self.write(path,[{k:v for k,v in row.items() if k!="text"} | {"messages":[
            {"role":"user","content":f"Pregunta ficticia {i}."},{"role":"assistant","content":f"Respuesta ficticia {i}."}]}
            for i,row in enumerate(self.rows)])
        initial_states=[]; original=training._checkpoint
        def checkpoint(*args,**kwargs):
            if args[5]["step"]==0:
                initial_states.append(({key:value.clone() for key,value in args[1].state_dict().items()},
                                       args[2].to_str(),copy.deepcopy(args[3].state_dict())))
            return original(*args,**kwargs)
        with patch.object(training,"_checkpoint",side_effect=checkpoint):
            result=training.train(path,self.root/"sft",initial_model=self.root/"parent",
                training_config={**self.config,"objective":"assistant_last_reply","max_steps":1})
        self.assertEqual(initial_states[0][1],tokenizer.to_str()); self.assertEqual(initial_states[0][2]["state"],{})
        for key,value in parent.state_dict().items(): self.assertTrue(torch.equal(value,initial_states[0][0][key]))
        self.assertEqual(result["trainingObjective"],"assistant_last_reply")
        self.assertEqual(result["lineage"]["parentModelId"],manifest["modelId"])
        with self.assertRaisesRegex(ValueError,"configuration mismatch"):
            training.train(path,self.root/"bad-resize",initial_model=self.root/"parent",model_config={"vocab_size":400},
                           training_config={**self.config,"objective":"assistant_last_reply"})


if __name__ == "__main__": unittest.main()
