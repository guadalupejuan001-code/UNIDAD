"""Invented conversation sources test persistence, bounds and isolation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ia_propia.context.store import ContextStore, ContextStoreError


def message(text, role="user"):
    return {"role": role, "content": text}


class ContextStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "private" / "context.sqlite3"
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.directory.cleanup()

    def store(self, *, product="pc", session="dialogue-1", **kwargs):
        result = ContextStore(self.path, product_id=product, session_id=session, **kwargs)
        self.stores.append(result)
        return result

    def test_batches_persist_complete_sources_and_repeated_real_turns(self):
        store = self.store()
        first = store.append([message("El taller abre el jueves."), message("El taller abre el lunes.")])
        second = store.append([message("El taller abre el lunes.")])
        self.assertEqual((first.inserted, first.total, second.inserted, second.total), (2, 2, 1, 3))
        self.assertEqual([source.turn_index for source in first.sources], [0, 1])
        self.assertNotEqual(first.sources[1].id, second.sources[0].id)
        self.assertEqual(first.sources[0].as_dict(), {
            "id": first.sources[0].id, "text": "El taller abre el jueves.",
            "role": "user", "turn_index": 0, "source": "conversation",
            "sha256": hashlib.sha256("El taller abre el jueves.".encode()).hexdigest(),
        })
        self.assertEqual(store._connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        store.close()
        loaded = self.store()
        self.assertEqual(loaded.count(), 3)
        self.assertEqual([source.text for source in loaded.recent()], [
            "El taller abre el jueves.", "El taller abre el lunes.", "El taller abre el lunes."])
        self.assertNotIn("path", loaded.stats())

    def test_snapshot_deduplicates_verifies_prefix_and_requires_explicit_reset(self):
        store = self.store()
        history = [message("Objetivo: organizar una visita."), message("¿Cuándo?", "assistant")]
        self.assertEqual(store.append_history(history).inserted, 2)
        self.assertEqual(store.append_history(history).inserted, 0)
        self.assertEqual(store.append_history(history[:1]).total, 2)
        self.assertEqual(store.append_history(history + [message("El lunes.")]).inserted, 1)
        with self.assertRaisesRegex(ContextStoreError, "prefix changed"):
            store.append_history([message("Objetivo: comprar algo.")])
        self.assertEqual(store.count(), 3)
        store.reset()
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.append_history([message("Una conversación nueva.")]).sources[0].turn_index, 0)

    def test_search_never_treats_assistant_text_as_facts_in_either_backend(self):
        for use_fts in (True, False):
            with self.subTest(use_fts=use_fts):
                store = self.store(session=str(use_fts), use_fts=use_fts)
                store.append([message("El observatorio abre el jueves."),
                              message("La biblioteca tiene un jardín.", "system"),
                              message("El observatorio abre el domingo. Dragones imaginarios.", "assistant")])
                self.assertEqual(store.search("dragones"), [])
                self.assertEqual([source.role for source in store.search("observatorio")], ["user"])
                self.assertEqual([source.role for source in store.search("jardín")], ["system"])
                self.assertEqual([source.role for source in store.recent()], ["user", "system"])
                self.assertEqual([source.role for source in store.recent(include_assistant=True)],
                                 ["user", "system", "assistant"])

    def test_scopes_and_reset_do_not_leak_or_remove_other_conversations(self):
        first = self.store()
        second = self.store(session="dialogue-2")
        phone = self.store(product="android")
        for store, day in ((first, "jueves"), (second, "lunes"), (phone, "sábado")):
            store.append([message(f"La biblioteca abre el {day}.")])
        self.assertEqual([source.text for source in first.search("biblioteca")], ["La biblioteca abre el jueves."])
        self.assertEqual([source.text for source in second.search("biblioteca")], ["La biblioteca abre el lunes."])
        self.assertEqual([source.text for source in phone.search("biblioteca")], ["La biblioteca abre el sábado."])
        first.reset()
        self.assertEqual(first.search("biblioteca"), [])
        self.assertEqual((second.count(), phone.count()), (1, 1))
        self.assertEqual(len(second.search("biblioteca")), 1)

    def test_fallback_scan_is_bounded_and_results_keep_complete_text(self):
        store = self.store(use_fts=False, fallback_scan_limit=3)
        store.append([message("Un telescopio antiguo."), message("Café con mañana."),
                      message("Café del taller."), message("Taller final.")])
        self.assertEqual(store.search("telescopio"), [])
        hits = store.search("cafe\u0301 TALLER", top_k=2)
        self.assertEqual([source.text for source in hits], ["Café del taller.", "Taller final."])
        self.assertEqual(store.search("café", max_chars=5), [])
        self.assertEqual([source.text for source in store.search("taller", max_chars=13)], ["Taller final."])
        self.assertEqual(store.search('" OR * NEAR(taller) --'), store.search("OR NEAR taller"))

    def test_enabling_fts_after_fallback_builds_old_index_and_tracks_all_writers(self):
        fallback = self.store(use_fts=False)
        fallback.append([message("Un cometa violeta.")])
        indexed = self.store()
        if not indexed.fts_enabled:
            self.skipTest("SQLite does not provide FTS5")
        self.assertEqual(len(indexed.search("cometa")), 1)
        fallback.append([message("Otro cometa azul.")])
        self.assertEqual(len(indexed.search("cometa")), 2)
        fallback.reset()
        self.assertEqual(indexed.search("cometa"), [])

    def test_fts_finds_old_rare_entity_beside_common_words_beyond_fallback_window(self):
        store = self.store(fallback_scan_limit=8)
        if not store.fts_enabled:
            self.skipTest("SQLite does not provide FTS5")
        store.append([message("El taller Lucerna abre el jueves."),
                      message("El taller Lucerna cambió al lunes.")])
        for _ in range(3):
            store.append([message(f"El taller corriente {index} abre el martes.") for index in range(64)])
        hits = store.search("¿Cuándo abre el taller Lucerna?", top_k=2)
        self.assertEqual({source.turn_index for source in hits}, {0, 1})
        self.assertEqual(hits[0].text, "El taller Lucerna abre el jueves.")
        # A current request omitting 'abre' gives the more recent correction priority.
        self.assertEqual(store.search("taller Lucerna", top_k=1)[0].turn_index, 1)
        self.assertEqual(store.search("ficticio taller Lucerna", top_k=1)[0].turn_index, 1)

    def test_first_storage_schema_upgrade_preserves_source_identity_and_hash(self):
        self.path.parent.mkdir()
        text = "La nota antigua conserva una brújula."
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        previous = hashlib.sha256(b"unidad-context-v1").digest()
        prefix_hash = hashlib.sha256(previous + b"user" + bytes.fromhex(text_hash)).hexdigest()
        with sqlite3.connect(self.path) as legacy:
            legacy.execute("""CREATE TABLE context_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,session_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
                role TEXT NOT NULL,text TEXT NOT NULL,sha256 TEXT NOT NULL,prefix_sha256 TEXT NOT NULL,
                UNIQUE(product_id,session_id,ordinal))""")
            legacy.execute("INSERT INTO context_sources VALUES(1,'pc','dialogue-1',0,'user',?,?,?)",
                           (text, text_hash, prefix_hash))
            try:
                legacy.execute("""CREATE VIRTUAL TABLE context_sources_fts
                    USING fts5(text,content='context_sources',content_rowid='source_id')""")
                legacy.execute("INSERT INTO context_sources_fts(context_sources_fts) VALUES('rebuild')")
            except sqlite3.OperationalError as exc:
                if "no such module" not in str(exc):
                    raise
        store = self.store()
        self.assertEqual(store.append_history([message(text)]).inserted, 0)
        source = store.search("brújula")[0]
        self.assertEqual((source.id, source.text, source.sha256), ("context:1", text, text_hash))
        store.append([message("La brújula cambió de sitio.")])
        self.assertEqual(len(store.search("brújula")), 2)

    def test_invalid_input_and_storage_failure_leave_batch_atomic(self):
        store = self.store()
        store.append([message("El museo está abierto.")])
        with self.assertRaises(ContextStoreError):
            store.append([message("Primero válido."), message("x" * 16_385)])
        self.assertEqual(store.count(), 1)
        store._connection.execute("""CREATE TRIGGER reject_test BEFORE INSERT ON context_sources
            WHEN new.text='fallo' BEGIN SELECT RAISE(ABORT,'fixture storage failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            store.append([message("Este registro debe revertirse."), message("fallo")])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.search("revertirse"), [])

    def test_parallel_batches_and_independent_connections_preserve_ordinals(self):
        first, second = self.store(), self.store()
        def write(index):
            store = first if index % 2 else second
            return store.append([message(f"Lote {index}, turno {turn}.") for turn in range(4)])
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(write, range(16)))
        self.assertEqual(sum(result.inserted for result in results), 64)
        self.assertEqual(first.count(), 64)
        self.assertEqual([source.turn_index for source in first.recent(limit=64)], list(range(64)))

    def test_validation_bounds_and_closed_handle(self):
        store = self.store()
        invalid_batches = [None, {}, [message("x")] * 65, [message("", "user")],
                           [message("x", "tool")], [message("\ud800")],
                           [{"role": "user", "content": "x", "path": "elsewhere"}]]
        for batch in invalid_batches:
            with self.subTest(batch_type=type(batch).__name__):
                with self.assertRaises(ContextStoreError):
                    store.append(batch)
        for query, kwargs in ((None, {}), ("x" * 4097, {}), ("\ud800", {}),
                              ("x", {"top_k": True}), ("x", {"top_k": 65}),
                              ("x", {"max_chars": 0})):
            with self.assertRaises(ContextStoreError):
                store.search(query, **kwargs)
        with self.assertRaises(ContextStoreError):
            store.recent(limit=65)
        with self.assertRaises(ContextStoreError):
            store.recent(include_assistant=1)
        store.close()
        store.close()
        with self.assertRaises(ContextStoreError):
            store.count()


if __name__ == "__main__":
    unittest.main()
