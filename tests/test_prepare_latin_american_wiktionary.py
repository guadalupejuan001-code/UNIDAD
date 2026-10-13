"""Regional-sense guards with original synthetic fixtures and no network calls.

Fixtures exercise parser semantics only. They are not training records, source
revisions, held-out dialogue cases, or quotations from external publications.
"""
import importlib.util
from pathlib import Path
import sys
import unittest


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_PATH = _EXAMPLES / "prepare_latin_american_wiktionary.py"
_SPEC = importlib.util.spec_from_file_location("prepare_latin_american_wiktionary", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_EXAMPLES))
try:
    _SPEC.loader.exec_module(helper)
finally:
    sys.path.remove(str(_EXAMPLES))


class LatinAmericanSenseParserTests(unittest.TestCase):
    def test_etymologies_disambiguate_repeated_sense_numbers_and_scope_notes(self):
        source = """=== Etimología 1 ===
==== {{sustantivo femenino|es}} ====
;1: Niño de corta edad.
{{ámbito|Argentina|nota=Cuyo|Bolivia|nota2=Tarija|Chile}}
=== Etimología 2 ===
==== {{sustantivo femenino|es}} ====
;1: Vehículo de transporte colectivo.
{{ámbito|Cuba|Puerto Rico|República Dominicana}}
"""
        senses = helper.regional_senses(source)
        self.assertEqual([(sense["etymology"], sense["senseNumber"])
                          for sense in senses], [(1, "1"), (2, "1")])
        self.assertEqual(senses[0]["scope"], [
            {"label": "Argentina", "qualifier": "Cuyo"},
            {"label": "Bolivia", "qualifier": "Tarija"},
            {"label": "Chile", "qualifier": ""},
        ])
        self.assertEqual([item["label"] for item in senses[1]["scope"]],
                         ["Cuba", "Puerto Rico", "República Dominicana"])
        rendered = helper.render("voz de fixture", senses)
        self.assertIn("Acepción 1.1", rendered)
        self.assertIn("Acepción 2.1", rendered)
        self.assertIn("Argentina (Cuyo); Bolivia (Tarija); Chile", rendered)

    def test_scope_and_usage_do_not_leak_to_following_sense_or_etymology(self):
        source = """=== Etimología 1 ===
==== {{sustantivo femenino|es}} ====
;1: Situación favorable por casualidad.
{{uso|coloquial}}
;2: Trabajo o empleo remunerado.
{{uso|coloquial}}
{{ámbito|México|Centroamérica|Perú}}
;3: Significado sin un ámbito explícito.
=== Etimología 2 ===
==== {{sustantivo femenino|es}} ====
;1: Hendidura profunda en la tierra.
{{ámbito|Colombia}}
"""
        senses = helper.regional_senses(source)
        self.assertEqual(len(senses), 2)
        self.assertEqual([item["label"] for item in senses[0]["scope"]],
                         ["México", "Centroamérica", "Perú"])
        self.assertEqual(senses[1]["scope"],
                         [{"label": "Colombia", "qualifier": ""}])
        self.assertEqual(senses[1]["usage"], [])

    def test_external_quotations_and_referenced_definitions_are_excluded(self):
        source = """==== {{sustantivo masculino|es}} ====
;1: Amigo de una persona.
{{ámbito|Venezuela}}
{{ejemplo|Un ejemplo original de la fuente sin cita.}}
{{ejemplo|Un texto externo de otra publicación.|autor=Una autora}}
{{ejemplo|Otro texto citado.<ref>Una publicación externa.</ref>}}
;2: Otra definición.<ref>Una obra externa.</ref>
{{ámbito|México}}
;3: {{DLE|voz}}
{{ámbito|Chile}}
"""
        senses = helper.regional_senses(source)
        self.assertEqual(len(senses), 1)
        self.assertEqual(senses[0]["examples"],
                         ["Un ejemplo original de la fuente sin cita."])

    def test_a_page_country_category_does_not_assign_scope_to_all_definitions(self):
        source = """==== {{sustantivo masculino|es}} ====
;1: Sentido general sin marca regional.
;2: Sentido peninsular de la palabra.
{{ámbito|España}}
;3: Sentido regional expresamente marcado.
{{ámbito|Ecuador}}
[[Categoría:ES:México]]
"""
        senses = helper.regional_senses(source)
        self.assertEqual([sense["senseNumber"] for sense in senses], ["3"])
        self.assertEqual(senses[0]["scope"],
                         [{"label": "Ecuador", "qualifier": ""}])

    def test_explicit_first_note_alias_is_not_silently_lost(self):
        # The parser admits nota1 as a named note. It must preserve its value
        # or reject the unresolved scope, rather than publish Argentina alone.
        source = """==== {{sustantivo femenino|es}} ====
;1: Persona de corta edad.
{{ámbito|Argentina|nota1=Cuyo|Bolivia|nota2=Tarija}}
"""
        senses = helper.regional_senses(source)
        if senses:
            self.assertEqual(senses[0]["scope"], [
                {"label": "Argentina", "qualifier": "Cuyo"},
                {"label": "Bolivia", "qualifier": "Tarija"},
            ])

    def test_orphan_or_conflicting_notes_do_not_broaden_a_qualified_scope(self):
        for scope in ("{{ámbito|Argentina|nota3=Cuyo}}",
                      "{{ámbito|Argentina|nota=Cuyo|nota1=Mendoza}}"):
            with self.subTest(scope=scope):
                source = ("==== {{sustantivo femenino|es}} ====\n"
                          ";1: Persona de corta edad.\n" + scope + "\n")
                self.assertEqual(helper.regional_senses(source), [])


if __name__ == "__main__":
    unittest.main()
