"""Reproduce one archived C7 dialogue; this is a regression, not acceptance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from ia_propia.conversation.runtime import ConversationEngine

    torch.set_num_threads(1)
    archive = ROOT / "docs/brain-nerva-20261013/reserved-c7.actual.json"
    prior = json.loads(archive.read_text())
    model_dir = ROOT / "models/conversation-reference-c7"
    members = ["config.json", "tokenizer.json", "weights.safetensors", "manifest.json"]

    def identities():
        return {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                for name in members}

    before = identities()
    engine = ConversationEngine(model_dir)
    history, turns = [], []
    for turn in prior["episodes"][0]["turns"]:
        history.append({"role": "user", "content": turn["user"]})
        actual = engine.reply(history, max_new_tokens=160, temperature=0, seed=42)
        turns.append({"number": turn["number"], "user": turn["user"], "actual": actual,
                      "matchesArchivedReply": actual == turn["actual"],
                      "historyMatchesArchive": history == turn["inputMessages"]})
        history.append({"role": "assistant", "content": actual["text"]})
    after = identities()
    report = {"purpose": "archived-reference-regression-not-new-acceptance",
              "modelId": engine.model_id, "archive": str(archive.relative_to(ROOT)),
              "archiveSha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "runnerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "artifactSha256": after, "artifactUnmodified": before == after,
              "matchesPublishedArtifact": after == prior["artifactSha256"],
              "historyUsesActualGeneratedReplies": True, "turns": turns,
              "trainingPerformed": False, "newConversationAcceptancePerformed": False,
              "conversationQualityImproved": False}
    path = Path(__file__).with_name("reference-regression.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not (report["artifactUnmodified"] and report["matchesPublishedArtifact"]
            and all(t["matchesArchivedReply"] and t["historyMatchesArchive"] for t in turns)):
        raise RuntimeError("C7 regression or identity mismatch; inspect the receipt")
    print("C7: four archived replies reproduced; artifact unchanged.")


if __name__ == "__main__":
    main()
