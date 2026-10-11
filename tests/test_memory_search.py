"""Lexical retrieval oracles use invented records, without any model runtime."""

from copy import deepcopy
import hashlib
import json
import unittest
import unicodedata

from ia_propia.memory_search import search_snapshot, validate_search


def snapshot(tasks):
    return {"workspaceId": "workspace_" + "0" * 32, "eventIds": {"one", "two"},
            "tasks": {task["id"]: task for task in tasks}}


def task(number, goal, context="", attempts=(), created_at=1):
    return {"id": "task_" + f"{number:032x}", "eventId": "event_" + f"{number:032x}",
            "createdAt": created_at, "goal": goal, "context": context, "attempts": list(attempts)}


def attempt(number, text, feedback=()):
    return {"id": "attempt_" + f"{number:032x}", "eventId": "event_" + f"{number + 100:032x}",
            "text": text, "accepted": feedback[-1][1] if feedback else None,
            "feedback": [{"text": value, "accepted": accepted,
                          "eventId": "event_" + f"{number + index + 200:032x}"}
                         for index, (value, accepted) in enumerate(feedback)]}


class MemorySearchTests(unittest.TestCase):
    def test_all_words_can_span_fields_but_cannot_span_tasks_or_partial_words(self):
        state = snapshot([task(1, "Preparar informe", "Para el taller solar"),
                          task(2, "Comprar pintura", "El informe llegará después")])
        found = search_snapshot(state, "INFORME taller")
        self.assertEqual([item["id"] for item in found["tasks"]], [task(1, "")["id"]])
        self.assertEqual([match["source"]["field"] for match in found["tasks"][0]["matches"]], ["goal", "context"])
        self.assertEqual(search_snapshot(state, "pintura taller")["matchCount"], 0)
        self.assertEqual(search_snapshot(state, "sol")["matchCount"], 0)
        self.assertEqual(search_snapshot(state, "cuadro")["matchCount"], 0)

    def test_only_current_acceptance_and_latest_feedback_are_searchable(self):
        rejected = attempt(1, "Cuerpo descartado", [("Aprobación obsoleta", True), ("Usa sábado", False)])
        accepted = attempt(2, "Entrega vigente", [("Primera antigua", False), ("Conserva viernes", True)])
        pending = attempt(3, "Cuerpo pendiente")
        state = snapshot([task(1, "Organizar entrega", attempts=[rejected, accepted, pending])])
        for excluded in ("descartado", "obsoleta", "antigua", "pendiente"):
            self.assertEqual(search_snapshot(state, excluded)["tasks"], [], excluded)
        source = search_snapshot(state, "sábado")["tasks"][0]["matches"][0]["source"]
        self.assertEqual(source["field"], "feedback.text")
        self.assertFalse(source["accepted"])
        self.assertEqual(source["eventId"], rejected["feedback"][-1]["eventId"])
        source = search_snapshot(state, "vigente")["tasks"][0]["matches"][0]["source"]
        self.assertEqual(source["field"], "attempt.text")
        self.assertTrue(source["accepted"])
        accepted["accepted"] = False
        accepted["feedback"].append({"text": "Ahora domingo", "accepted": False, "eventId": "new-event"})
        self.assertEqual(search_snapshot(state, "vigente")["tasks"], [])
        self.assertEqual(search_snapshot(state, "viernes")["tasks"], [])
        self.assertEqual(search_snapshot(state, "domingo")["matchCount"], 1)

    def test_unicode_casefold_preserves_original_offsets_and_exact_excerpt(self):
        text = "Inicio " + "á" * 180 + " STRASSE, café 東京. " + "é" * 180
        state = snapshot([task(1, text)])
        result = search_snapshot(state, "straße CAFÉ 東京")
        for match in result["tasks"][0]["matches"]:
            source = match["source"]
            self.assertEqual(unicodedata.normalize("NFC", text[source["start"]:source["end"]]).casefold(), match["term"])
            self.assertEqual(match["excerpt"], text[match["excerptStart"]:match["excerptEnd"]])
            self.assertLessEqual(len(match["excerpt"]), 160)
            self.assertEqual(source["textSha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertIsNone(source["accepted"])
        self.assertEqual(search_snapshot(state, "cafe")["matchCount"], 0)
        self.assertEqual(validate_search("Café café CAFÉ"), ("café",))

    def test_combining_marks_keep_accents_and_canonical_equivalence_with_original_spans(self):
        nfd = "Pedido cafe\u0301 y man\u0303ana."
        nfc = "Pedido café y mañana."
        plain = "Pedido cafe y manana."
        state = snapshot([task(1, nfd), task(2, nfc), task(3, plain)])
        for query in ("café mañana", "cafe\u0301 man\u0303ana"):
            result = search_snapshot(state, query)
            self.assertEqual([item["id"] for item in result["tasks"]], [task(1, "")["id"], task(2, "")["id"]])
            for item in result["tasks"]:
                original = state["tasks"][item["id"]]["goal"]
                for match in item["matches"]:
                    source = match["source"]
                    span = original[source["start"]:source["end"]]
                    self.assertEqual(unicodedata.normalize("NFC", span).casefold(), match["term"])
                    self.assertEqual(match["excerpt"], original[match["excerptStart"]:match["excerptEnd"]])
                    self.assertEqual(source["textSha256"], hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual([item["id"] for item in search_snapshot(state, "cafe manana")["tasks"]], [task(3, "")["id"]])
        self.assertEqual(validate_search("café cafe\u0301 CAFÉ"), ("café",))
        # Mn, Mc and Me continue the original word even without an NFC composite.
        for mark in ("\u0301", "\u0903", "\u20dd"):
            with self.subTest(category=unicodedata.category(mark)):
                original = "ab" + mark + "cd"
                marked = snapshot([task(1, original), task(2, "ab cd")])
                self.assertEqual(search_snapshot(marked, "ab")["tasks"][0]["id"], task(2, "")["id"])
                hit = search_snapshot(marked, original)["tasks"][0]["matches"][0]
                self.assertEqual((hit["source"]["start"], hit["source"]["end"]), (0, len(original)))

    def test_results_are_stably_bounded_and_detached_from_source(self):
        state = snapshot([task(number, "Informe " + "x" * 400, created_at=number // 2)
                          for number in range(1, 26)])
        before = deepcopy(state)
        result = search_snapshot(state, "informe", limit=2)
        self.assertEqual([item["id"] for item in result["tasks"]], [task(24, "")["id"], task(25, "")["id"]])
        self.assertEqual(result["matchCount"], 25)
        self.assertTrue(result["truncated"])
        self.assertTrue(all(item["goalTruncated"] for item in result["tasks"]))
        self.assertTrue(all(len(item["goalExcerpt"]) <= 160 for item in result["tasks"]))
        result["tasks"][0]["matches"][0]["source"]["field"] = "changed"
        result["tasks"][0]["goalExcerpt"] = "changed"
        self.assertEqual(state, before)
        self.assertEqual(search_snapshot(state, "informe")["limit"], 10)

    def test_query_types_bytes_terms_and_limits_fail_without_coercion(self):
        for query, limit in ((None, 10), ("", 10), ("   ", 10), ("!!!", 10), ("a\ud800", 10),
                             ("é" * 257, 10), (" ".join(f"word{n}" for n in range(17)), 10),
                             ("informe", True), ("informe", 0), ("informe", 21), ("informe", "2")):
            with self.subTest(query=repr(query), limit=limit), self.assertRaises(ValueError):
                validate_search(query, limit)
        self.assertEqual(validate_search("é" * 256, 20), ("é" * 256,))

    def test_worst_bounded_unicode_result_fits_brain_response_budget(self):
        words = [chr(0x1D400 + number) * 3 for number in range(16)]
        query = " ".join(words)
        goal = " ".join(word + " " + "界" * 200 for word in words)
        result = search_snapshot(snapshot([task(number, goal) for number in range(20)]), query, 20)
        self.assertEqual(len(result["tasks"]), 20)
        self.assertTrue(all(len(item["matches"]) == 16 for item in result["tasks"]))
        self.assertLess(len(json.dumps(result, ensure_ascii=False).encode()), 1_048_576)


if __name__ == "__main__":
    unittest.main()
