"""Smoke-check the selected published weights and retain unedited model outputs.

This is technical integration evidence, never a conversational acceptance test.
All prompts and notes here are fictional development examples, outside TEST.
"""

import argparse
import json
from pathlib import Path
import tempfile


def verify(config_path):
    import torch
    from ia_propia.sovengi import Sovengi
    from ia_propia.sovengi.__main__ import _request

    torch.set_num_threads(1)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config["projectRoot"] = str((config_path.parent / config["projectRoot"]).resolve())
    with tempfile.TemporaryDirectory() as temp:
        config["brain"]["state_dir"] = str(Path(temp) / "state")
        temporary_config = Path(temp) / "config.json"
        temporary_config.write_text(json.dumps(config))
        app = Sovengi(temporary_config)
        try:
            session, other = app.connect(), app.connect("isolated")
            engine = session.models.conversation_engine
            assert engine is other.models.conversation_engine
            assert engine.model_id == config["brain"]["expected_model_id"]
            note = _request(session, "memory.task.add", {"goal": "El taller Lucerna abre el jueves."})
            count = _request(session, "memory.task.status", {})["eventCount"]
            cases = ["Hola. ¿Cómo te llamas?", "¿Qué día abre el taller Lucerna?",
                     "El faro Lumbre está en Puerto Cerezo. ¿Dónde está el faro Lumbre?"]
            outputs = []
            for prompt in cases:
                result = _request(session, "conversation.reply", {"messages": [{"role": "user", "content": prompt}],
                    "max_new_tokens": 48, "temperature": 0, "seed": 42})
                if config["brain"].get("conversation_ensemble") is not None:
                    trace = result["ensemble"]
                    assert trace["minimumCandidatesSatisfied"]
                    assert len(trace["candidates"]) >= 2
                    assert len(set(trace["modelIdsExecuted"])) >= 2
                    assert trace["selectedModelId"] == result["modelId"]
                    assert any(candidate.get("auxiliary") for candidate in trace["candidates"])
                    assert trace["cache"]["residentCount"] <= config["brain"]["conversation_ensemble"]["max_resident"]
                outputs.append({"prompt": prompt, "result": result})
            assert note["id"] in outputs[1]["result"]["memoryContext"]["taskIds"]
            assert _request(session, "memory.task.status", {})["eventCount"] == count
            isolated = _request(other, "conversation.reply", {"messages": [{"role": "user", "content": cases[1]}],
                "max_new_tokens": 1, "temperature": 0})
            assert isolated["memoryContext"]["taskIds"] == []
            return {"product": "SOVENGI", "modelId": engine.model_id,
                "parameterCount": engine.parameter_count, "modelConfig": engine.manifest["config"],
                "integrationVerified": True, "conversationQualityValidated": False,
                "productionAccepted": False, "trainingExecuted": False,
                "fixtureScope": "fictional prompts; real selected published weights; no reserved TEST",
                "ensembleConfigured": config["brain"].get("conversation_ensemble") is not None,
                "outputs": outputs}
        finally:
            app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gnosis.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
