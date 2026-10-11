"""Independent semantic checks for fictional natural dialogue authoring.

The semantic oracle consumes only user prose.  It never consults an authoring
world, event list, template identifier, answer renderer or model response.
Manual adversarial fixtures test the oracle itself before any generated data
is checked. The fixed tokenizer is the public, unchanged project tokenizer;
the suite does not depend on a private training workspace.
"""

from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIXED_TOKENIZER = ROOT / "models/conversation-basics-v4/tokenizer.json"
FOLDS = {"train", "validation", "test"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_prompt(messages):
    return tuple((message["role"], " ".join(unicodedata.normalize(
        "NFKC", message["content"]).casefold().split())) for message in messages)


def prose(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def literal(text):
    return prose(text).strip(" .!?¿¡,;:")


@dataclass(frozen=True)
class Requirement:
    """A required visible fact; no fields are taken from authoring metadata."""

    slot: str
    value: str | None
    obsolete: tuple[str, ...] = ()


class UserProseOracle:
    """Track facts in user statements, independently of the authoring helper.

    Named slots deliberately remain separate: selecting an alias does not
    establish a real name, and changing one object never changes its neighbor.
    Questions read state but cannot write it.  Corrections replace precisely
    the stated slot and retain obsolete values for negative target checks.
    """

    def __init__(self):
        self.facts = {}
        self.history = defaultdict(set)

    def remember(self, slot, value):
        value = literal(value)
        self.facts[slot] = value
        self.history[slot].add(value)
        return self.require(slot)

    def require(self, slot):
        current = self.facts.get(slot)
        return Requirement(slot, current, tuple(sorted(self.history[slot] - {current})))

    def observe(self, user):
        """Read literal Spanish statements; metadata never enters this path."""
        text = prose(user)
        # Requests have no authority to supply new facts.  They must be
        # recognized before the more general declaration grammar.
        request = self.question(text, optional=True)
        if request is not None:
            return request
        if any(mark in text for mark in ('«', '»', '“', '”', '"')):
            raise AssertionError("Quoted prose is not an assertion: " + user)
        if re.search(r"mi [\wáéíóúüñ]+ no es", text) and "; es " not in text:
            raise AssertionError("Negating a value does not establish its replacement: " + user)
        found = []
        name_patterns = (
            r"(?:hola[,.] |buenas, |para presentarme: |me equivoqué: |corrijo: |en realidad )?(?:me llamo|soy|mi nombre(?: correcto)? es) ([^.;?]+)",
            r"puedes conocerme por mi nombre, (.+)",
            r"(.+?) es mi nombre(?:[.;]|$)",
            r"corrijo mi nombre: (.+)", r"antes dije mal mi nombre; es (.+)",
            r"cambia mi nombre a (.+)", r"no me llamo así, sino (.+)",
            r"quédate con mi nombre correcto: (.+)",
            r"aquí ([^,]+), desde ", r"me presento: ([^,]+), ahora en ",
        )
        for pattern in name_patterns:
            match = re.match(pattern, text)
            if match:
                value = re.split(r" y (?:ahora )?estoy en | y me encuentro en ", match[1])[0]
                found.append(self.remember("name", value))
                break
        alias_patterns = (
            r"(?:hola, )?llámame (.+)", r"prefiero que me llames (.+)",
            r"para esta charla, mi apodo es (.+)", r"puedes decirme (.+), como apodo",
            r"aquí quiero usar el apodo (.+)", r"me gustaría que me dijeras (.+)",
            r"me gusta que me llamen (.+)", r"usa (.+) para dirigirte a mí",
        )
        for pattern in alias_patterns:
            match = re.fullmatch(pattern + r"[.!]?", text)
            if match:
                found.append(self.remember("alias", match[1]))
                break
        city_patterns = (
            r"(?:estoy|me encuentro) en ([^.;]+)", r"te escribo desde ([^.;]+)",
            r"estoy pasando el día en (.+)", r"estoy aquí, en (.+)",
            r"me moví a (.+)", r"mi lugar actual es (.+)",
            r"(?:aquí [^,]+, desde|me presento: [^,]+, ahora en) (.+)",
            r"(?:ahora )?vivo en (.+)", r"mi ciudad(?: actual)? es (.+)",
        )
        for pattern in city_patterns:
            match = re.search(pattern, text)
            if match and not re.search(r"dice que|(?:no|nunca) (?:vivo|estoy|me encuentro) en", text):
                found.append(self.remember("city", match[1]))
                break
        topic_patterns = (
            r"(?:hola, )?(?:quiero|me apetece|prefiero) hablar de (.+)",
            r"tengo en mente el proyecto (.+)", r"hoy traigo el tema (.+)",
            r"hablemos del proyecto (.+)", r"quería contarte algo sobre (.+)",
            r"el tema que me interesa es (.+)", r"propongo conversar sobre (.+)",
            r"(?:ahora |cambié de idea: )?hablemos de (.+)", r"cambio de tema: (.+)",
            r"dejemos eso; me interesa (.+)", r"el tema actual será (.+)",
            r"mejor conversemos sobre (.+)", r"quiero pasar al proyecto (.+)",
        )
        for pattern in topic_patterns:
            match = re.fullmatch(pattern + r"[.!]?", text)
            if match:
                found.append(self.remember("topic", match[1]))
                break
        project = re.fullmatch(r"(?:trabajo en(?: el proyecto)?|quiero trabajar en(?: el proyecto)?|mi proyecto(?: actual)? es) (.+)", text)
        if project:
            found.append(self.remember("project", project[1]))
        # Color relations bind each adjective to its own explicit object.
        # Ellipsis after a semicolon is resolved only by its named subject.
        color_text = re.sub(r"^(?:corrijo: |me equivoqué: |por cierto, |te cuento que |dos colores: |para recordar: |quédate con este color: |en realidad, |antes dije mal el color; |ahora )", "", text)
        color_patterns = (
            r"\b(?:mi|el|la|un|una) ([\wáéíóúüñ]+)(?: que uso)? (?:es |de color )?([\wáéíóúüñ]+)(?=[.;,]|$| y )",
            r"mi ([\wáéíóúüñ]+), ([\wáéíóúüñ]+)",
            r"(?:el color (?:correcto )?de|el de) mi ([\wáéíóúüñ]+)(?: es|,) ([\wáéíóúüñ]+)",
            r"mi ([\wáéíóúüñ]+) no es así; es ([\wáéíóúüñ]+)",
            r"cambia el color de mi ([\wáéíóúüñ]+) a ([\wáéíóúüñ]+)",
        )
        color_values = {"azul", "verde", "rojo", "roja", "amarillo", "amarilla", "negro", "negra",
                        "blanco", "blanca", "gris", "ocre", "rosa", "violeta", "naranja", "marrón",
                        "morado", "morada", "beige"}
        colors = {}
        for pattern in color_patterns:
            for match in re.finditer(pattern, color_text):
                if match[2] in color_values:
                    colors[match[1]] = match[2]
        found.extend(self.remember("color:" + item, color) for item, color in colors.items())
        if re.search(r"(?:sólo|solo) quiero (?:que me escuches|hablar)|nada de consejos|hablar sin preparar un plan|sin buscar soluciones|sin recibir propuestas|me basta con que me escuches|quiero que escuches, sin consejos", text):
            found.append(self.remember("intent", "listen"))
        elif re.search(r"propón un primer paso|idea sencilla para empezar|propongas algo práctico|busco un paso pequeño|empezar con una propuesta|quiero un primer paso|idea práctica para comenzar|sugieras cómo empezar|(?:propón|proponme) (?:un|el) (?:primer|siguiente) paso", text):
            found.append(self.remember("intent", "act"))
        if found:
            return found
        if text in {"hola.", "hola", "buenas.", "buenas", "gracias.", "gracias"}:
            return [Requirement("social", None)]
        raise AssertionError("Independent oracle does not recognize user prose: " + user)

    def question(self, text, optional=False):
        query = literal(text)
        requests = (
            ("identity", r"cómo te llamas|quién eres|dime tu nombre|cuál es tu nombre|cómo puedo llamarte|preséntate con tu nombre|qué nombre usas tú|con quién estoy hablando"),
            ("alias", r"cómo quiero que me llames|qué apodo elegí|dime el apodo que uso aquí|qué nombre prefiero para esta charla|recuerdas cómo quiero que me digas|repite mi apodo, por favor|cómo te pedí que me llamaras|con qué apodo me tratarás|cuál es mi apodo"),
            ("name", r"cómo me llamo|cuál es mi nombre(?: real)?|dime mi nombre|recuerdas mi nombre|qué nombre mío te dije|con qué nombre me presenté|quiero saber qué nombre mío recuerdas|puedes repetir mi nombre real"),
            ("city", r"dónde estoy ahora|en qué lugar estoy|dime dónde me encuentro|cuál es mi lugar actual|desde dónde te escribo ahora|dónde dije que estaba|recuerdas mi ubicación actual|repite el lugar donde estoy|dónde vivo|cuál es mi ciudad(?: actual)?|recuerdas mi ciudad"),
            ("topic", r"de qué hablamos ahora|cuál es el tema actual|dime qué proyecto elegí|qué tema tengo en mente ahora|recuerdas de qué quería hablar|repite el tema que escogí|sobre qué estamos conversando|qué proyecto estamos tratando"),
            ("project", r"en qué proyecto trabajo|cuál es mi proyecto(?: actual)?|en qué quiero trabajar"),
            ("intent", r"qué quiero de esta charla ahora|te pedí escuchar o proponer|cómo quiero que me acompañes ahora|dime qué te pedí hacer ahora|qué prefiero en este momento|quieres recordar mi petición actual|cuál fue mi última petición para la charla|qué te toca hacer según lo que pedí"),
        )
        for slot, pattern in requests:
            if re.fullmatch(pattern, query):
                return [Requirement("identity", "unidad")] if slot == "identity" else [self.require(slot)]
        if re.fullmatch(r"qué sabes por lo que te conté|relaciona brevemente lo que te dije|qué recuerdas de mí y del tema|resume mi nombre, lugar y tema si los sabes|une los datos que compartí, sin inventar|dime los datos actuales de esta charla|puedes conectar lo que ya te conté|haz un resumen corto de los hechos", query):
            return [self.require(slot) for slot in ("name", "city", "topic") if slot in self.facts]
        if re.match(r"(?:de qué color|qué color|dime el color|cómo es el color|recuerdas el color|repite el color|cuál es el color)", query):
            items = re.findall(r"(?:mi|el|la|un|una) ([\wáéíóúüñ]+)", query)
            if items:
                return [self.require("color:" + items[-1])]
        if optional:
            return None
        raise AssertionError("Independent oracle does not recognize question: " + text)

    def assert_grounded(self, case, reply):
        """Any extra factual claim must also have independent user support."""
        text = prose(reply)
        patterns = {
            "name": r"(?:te llamas|hola,) ([^.!?;,]+)",
            "alias": r"(?:te llamaré|quieres que te llame|tu apodo es) ([^.!?;,]+)",
            "city": r"(?:estás en|vives en|tu ciudad es) ([^.!?;,]+)",
            "topic": r"(?:hablemos de|hablamos de|podemos anotar una idea para) ([^.!?;,]+)",
        }
        for slot, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                case.assertIn(slot, self.facts, "Assistant invented an undisclosed " + slot)
                case.assertEqual(literal(match[1]), self.facts[slot])
        for match in re.finditer(r"(?:el|la) ([\wáéíóúüñ]+) es ([\wáéíóúüñ]+)", text):
            slot = "color:" + match[1]
            case.assertIn(slot, self.facts, "Assistant invented an object's color")
            case.assertEqual(match[2], self.facts[slot])

    @staticmethod
    def assert_reply(case, required, reply):
        """Check relation and polarity, rather than mere target inclusion."""
        text = prose(reply)
        if required.slot == "social":
            case.assertRegex(text, r"hola|buenas|te leo|gracias|de nada")
            return
        if required.slot == "intent":
            if required.value is None:
                case.assertRegex(text, r"(?:no|todavía no|aún no).*(?:dicho|indicado|mencionado|sé|conozco)")
                return
            if required.value == "listen":
                case.assertRegex(text, r"te escucho|quieres que te escuche|escucharte")
                case.assertNotRegex(text, r"(?:primero|primer paso|te propongo)")
            else:
                case.assertRegex(text, r"(?:primer|siguiente) paso|proponga|proponer|proponerte|idea|quieres empezar")
            return
        relations = {
            "identity": r"(?:soy|me llamo|mi nombre es)",
            "name": r"(?:te llamas|tu nombre(?: real)? es|hola,)",
            "alias": r"(?:te llamaré|quieres que te llame|tu apodo es)",
            "city": r"(?:estás en|vives en|tu ciudad(?: actual)? es)",
            "project": r"(?:trabajas en|quieres trabajar en|tu proyecto(?: actual)? es)",
            "topic": r"(?:hablemos de|hablamos de|el tema(?: actual)? es|estamos hablando de)",
            "count": r"(?:tienes)",
        }
        if required.slot.startswith("color:"):
            subject = re.escape(required.slot.split(":", 1)[1])
            relation = r"(?:el|la|tu) " + subject + r" (?:es|es de color)"
        else:
            relation = relations[required.slot]
        if required.value is None:
            case.assertRegex(text, r"(?:no|todavía no|aún no).*(?:dicho|indicado|mencionado|sé|conozco)")
            # A missing-data reply must not simultaneously assert a fact.
            case.assertNotRegex(text, relation + r" ")
            return
        value = re.escape(required.value)
        case.assertRegex(text, relation + r" " + value + r"(?:\b|[.!])")
        case.assertNotRegex(text, r"(?:no|nunca|jamás) (?:" + relation + r") " + value)
        case.assertNotIn("?", text)
        case.assertNotRegex(text, r"[«\"“].*" + value)
        for old in required.obsolete:
            case.assertNotRegex(text, relation + r" " + re.escape(old) + r"(?=[.!?;,]|$| y )")
        # Catch a second, contradictory assertion even when its value was
        # never an obsolete value for this slot (for example a neighbor's
        # color).  Merely including the correct value cannot satisfy this.
        for clause in re.split(r" y (?=(?:el|la|tu|te|vives|estás|soy|me|mi) )", text):
            for match in re.finditer(relation + r" ([^.!?;,]+)", clause):
                asserted = literal(match[1])
                if required.slot == "count":
                    asserted = asserted.split(" ", 1)[0]
                case.assertEqual(asserted, required.value)


class IndependentOracleTests(unittest.TestCase):
    def test_manual_name_correction_keeps_alias_and_model_identity_separate(self):
        oracle = UserProseOracle()
        oracle.observe("Me llamo Lucía.")
        oracle.observe("Llámame Lu.")
        oracle.observe("Corrijo: me llamo Luciana.")
        requirements = [oracle.observe(question)[0] for question in
                        ("¿Cómo me llamo?", "¿Cuál es mi apodo?", "¿Cómo te llamas?")]
        self.assertEqual([item.value for item in requirements], ["luciana", "lu", "unidad"])
        for required, reply in zip(requirements, ("Te llamas Luciana.", "Tu apodo es Lu.", "Soy Unidad.")):
            oracle.assert_reply(self, required, reply)
        for bad in ("Te llamas Lucía.", "Te llamas Lu.", "No te llamas Luciana.",
                    "Dijiste «Luciana», pero te llamas Lucía.", "¿Te llamas Luciana?"):
            with self.subTest(reply=bad), self.assertRaises(AssertionError):
                oracle.assert_reply(self, requirements[0], bad)

    def test_manual_object_correction_preserves_the_neighbor(self):
        oracle = UserProseOracle()
        oracle.observe("Mi mochila es verde.")
        oracle.observe("Mi bicicleta es azul.")
        oracle.observe("Ahora mi mochila es roja.")
        bag = oracle.observe("¿De qué color es mi mochila?")[0]
        bicycle = oracle.observe("¿De qué color es mi bicicleta?")[0]
        self.assertEqual((bag.value, bicycle.value), ("roja", "azul"))
        oracle.assert_reply(self, bag, "La mochila es roja.")
        oracle.assert_reply(self, bicycle, "La bicicleta es azul.")
        for bad in ("La mochila es verde.", "La mochila es azul.", "La bicicleta es roja.",
                    "No la mochila es roja.", "La mochila es roja. La mochila es azul.",
                    "La mochila es roja y la mochila es azul."):
            with self.subTest(reply=bad), self.assertRaises(AssertionError):
                oracle.assert_reply(self, bag, bad)

    def test_questions_quotes_negations_and_assistant_prose_do_not_supply_user_facts(self):
        oracle = UserProseOracle()
        unknown = oracle.observe("¿Cómo me llamo?")[0]
        self.assertIsNone(unknown.value)
        oracle.assert_reply(self, unknown, "Todavía no has dicho tu nombre.")
        with self.assertRaises(AssertionError):
            oracle.assert_reply(self, unknown, "Te llamas Ana.")
        for text in ("El ejemplo dice «me llamo Ana».", "Mi mochila no es roja.",
                     "El asistente dice que vivo en Málaga."):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                oracle.observe(text)
        self.assertEqual(oracle.facts, {})
        oracle.observe("Vivo en Sevilla.")
        oracle.observe("Trabajo en el proyecto Bilbao.")
        city = oracle.observe("¿Dónde vivo?")[0]
        self.assertEqual(city.value, "sevilla")
        with self.assertRaises(AssertionError):
            oracle.assert_reply(self, city, "Vives en Bilbao.")

    def test_summary_cannot_add_an_undisclosed_attribute(self):
        oracle = UserProseOracle()
        oracle.observe("Me llamo Lina.")
        oracle.observe("Hablemos del proyecto Faro.")
        oracle.assert_grounded(self, "Te llamas Lina. Hablamos de Faro.")
        with self.assertRaises(AssertionError):
            oracle.assert_grounded(self, "Te llamas Lina. Estás en Murcia. Hablamos de Faro.")

    def test_complete_fact_value_can_extend_a_previous_value(self):
        oracle = UserProseOracle()
        oracle.observe("Hablemos del proyecto Faro.")
        required = oracle.observe("Cambio de tema: Faro Inicial.")[0]
        self.assertEqual(required.value, "faro inicial")
        oracle.assert_reply(self, required, "Hablemos de Faro Inicial.")
        with self.assertRaises(AssertionError):
            oracle.assert_reply(self, required, "Hablemos de Faro.")


@unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None and FIXED_TOKENIZER.exists(),
                     "Natural-diversity checks require tokenizers and the authorized fixed tokenizer")
class NaturalDiversityConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from examples import build_natural_diversity_conversations
        from ia_propia.conversation.data import load_corpus
        from tokenizers import Tokenizer

        cls.generator = build_natural_diversity_conversations
        cls.options = {"seed": 1813, "worlds": 80, "context_length": 256}
        cls.tokenizer_raw = FIXED_TOKENIZER.read_bytes()
        cls.tokenizer = Tokenizer.from_str(cls.tokenizer_raw.decode("utf-8"))
        cls.candidates = cls.generator.generate(seed=1813, worlds=80)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        cls.output, cls.manifest = root / "natural.jsonl", root / "manifest.json"
        cls.report = cls.generator.build(FIXED_TOKENIZER, cls.output, cls.manifest, **cls.options)
        cls.records = load_corpus(cls.output)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def build(self, output=None, manifest=None, **changes):
        return self.generator.build(FIXED_TOKENIZER, output or self.root / "natural.jsonl",
                                    manifest or self.root / "manifest.json", **{**self.options, **changes})

    def test_every_candidate_target_follows_only_visible_user_facts(self):
        covered = Counter()
        for row in self.candidates:
            oracle = UserProseOracle()
            for message in row["messages"]:
                if message["role"] == "user":
                    required = oracle.observe(message["content"])
            reply = row["messages"][-1]["content"]
            with self.subTest(record=row["id"]):
                for fact in required:
                    oracle.assert_reply(self, fact, reply)
                    covered[fact.slot.split(":", 1)[0]] += 1
                oracle.assert_grounded(self, reply)
        self.assertLessEqual({"name", "alias", "city", "topic", "identity", "color", "intent"}, set(covered))

    def test_complete_worlds_prefixes_and_every_paraphrase_keep_their_fold(self):
        groups, first_prompts, full_prompts = (defaultdict(set) for _ in range(3))
        flows, styles, full_rows = defaultdict(set), defaultdict(set), {}
        identifiers, messages = set(), set()
        for row in self.records:
            source = row["source"]
            self.assertNotIn(row["id"], identifiers)
            identifiers.add(row["id"])
            full_key = canonical(row["messages"])
            self.assertNotIn(full_key, messages)
            messages.add(full_key)
            groups[row["groupId"]].add(row["split"])
            first_prompts[normalized_prompt(row["messages"][:1])].add(row["split"])
            full_prompts[normalized_prompt(row["messages"][:-1])].add(row["split"])
            flows[source["flowId"]].add(row["split"])
            self.assertEqual(source["kind"], "authored-synthetic")
            self.assertIs(source["fictional"], True)
            self.assertIs(source["privateConversationUsed"], False)
            self.assertIs(source["modelResponsesUsed"], False)
            self.assertEqual(len(row["messages"]), 2 * source["prefixPairs"])
            self.assertEqual([message["role"] for message in row["messages"]],
                             ["user", "assistant"] * source["prefixPairs"])
            self.assertEqual(len(source["events"]), source["prefixPairs"])
            self.assertEqual(len(source["paraphrases"]), source["prefixPairs"])
            self.assertTrue(all(message["content"].strip() for message in row["messages"]))
            for event, style in zip(source["events"], source["paraphrases"]):
                styles[(event, row["split"])].add(style)
            if source["prefixPairs"] == source["fullFlowPairs"]:
                full_rows[(row["groupId"], source["flowId"])] = row
        self.assertEqual(len(groups), 80)
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))
        self.assertTrue(all(len(folds) == 1 for folds in first_prompts.values()))
        self.assertTrue(all(len(folds) == 1 for folds in full_prompts.values()))
        self.assertEqual(len(flows), 12)
        self.assertTrue(all(folds == FOLDS for folds in flows.values()))
        self.assertTrue(all(values == set(range(8)) for values in styles.values()))
        self.assertEqual(len(full_rows), 80 * 12)
        for row in self.records:
            source = row["source"]
            full = full_rows[(row["groupId"], source["flowId"])]
            self.assertEqual(row["messages"], full["messages"][:len(row["messages"])])
        self.assertEqual(Counter(next(iter(folds)) for folds in groups.values()),
                         {"train": 64, "validation": 8, "test": 8})

    def test_literal_values_have_one_world_attribute_owner_and_both_roles_vary_in_length(self):
        owners, role_lengths = defaultdict(set), defaultdict(set)
        for row in self.candidates:
            oracle = UserProseOracle()
            for message, event in zip(row["messages"][::2], row["source"]["events"]):
                requirements = oracle.observe(message["content"])
                for required in requirements:
                    if required.slot in {"name", "alias", "city", "topic"} and required.value is not None:
                        owners[required.value].add((row["groupId"], required.slot))
                        if event.startswith(("introduce_", "correct_")):
                            role = "corrected" if event.startswith("correct_") else "initial"
                            role_lengths[(required.slot, role)].add(len(required.value.split()))
        self.assertTrue(all(len(value_owners) == 1 for value_owners in owners.values()))
        for slot in ("name", "city", "topic"):
            for role in ("initial", "corrected"):
                with self.subTest(attribute=slot, role=role):
                    self.assertEqual(role_lengths[(slot, role)], {1, 2, 3})

    def test_real_trainer_teaches_final_reply_once_and_preserves_the_full_context(self):
        from ia_propia.conversation import training
        from ia_propia.conversation.tokenizer import ASSISTANT_ID, EOS_ID, IGNORE_INDEX

        with patch.object(training, "_torch", side_effect=AssertionError("No model backend is permitted")):
            encoded = training._encoded(self.records, self.tokenizer, 256)
        for fold in FOLDS:
            originals = [row for row in self.records if row["split"] == fold]
            self.assertEqual(len(encoded[fold]), len(originals))
            for row, (inputs, labels) in zip(originals, encoded[fold]):
                with self.subTest(record=row["id"]):
                    self.assertLessEqual(len(inputs), 256)
                    self.assertEqual(len(inputs), len(labels))
                    final = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
                    self.assertEqual(labels[:final], [IGNORE_INDEX] * final)
                    expected = self.tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False).ids + [EOS_ID]
                    self.assertEqual([label for label in labels if label != IGNORE_INDEX], expected)
                    previous = row["messages"][:-2]
                    for message in previous:
                        span = self.tokenizer.encode(message["content"], add_special_tokens=False).ids
                        self.assertTrue(any(inputs[index:index + len(span)] == span
                                            for index in range(final - len(span) + 1)))

    def test_build_is_reproducible_and_retains_every_complete_fitting_dialogue(self):
        from ia_propia.conversation.data import load_corpus
        from ia_propia.conversation.tokenizer import dialogue_tokens

        report = self.build()
        self.assertEqual(report, self.report)
        self.assertEqual((self.root / "natural.jsonl").read_bytes(), self.output.read_bytes())
        self.assertEqual((self.root / "manifest.json").read_bytes(), self.manifest.read_bytes())
        self.assertEqual(FIXED_TOKENIZER.read_bytes(), self.tokenizer_raw)
        self.assertEqual(report["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(report["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer_raw).hexdigest())
        self.assertIs(report["tokenizer"]["refitted"], False)
        self.assertIs(report["authoring"]["dialoguesTruncated"], False)
        self.assertIs(report["authoring"]["inferenceUsesAuthoringState"], False)
        self.assertEqual(report["authoring"]["completeFlowsExcluded"], 0)
        self.assertEqual(json.loads(self.manifest.read_text()), report)
        self.assertEqual(load_corpus(self.root / "natural.jsonl"), self.records)
        candidates = {row["id"]: row for row in self.candidates}
        for row in self.records:
            self.assertEqual(row, candidates[row["id"]])
            self.assertLessEqual(len(dialogue_tokens(row["messages"], self.tokenizer)[0]), 257)
        self.assertEqual(len(self.records), report["records"])
        self.assertEqual(set(self.root.iterdir()), {self.root / "natural.jsonl", self.root / "manifest.json"})
        self.assertEqual(self.generator.generate(seed=1813, worlds=80), self.candidates)
        self.assertNotEqual(self.generator.generate(seed=1814, worlds=80), self.candidates)

    def test_outputs_aliases_and_competing_writer_cannot_replace_evidence(self):
        for name in ("natural.jsonl", "manifest.json"):
            with self.subTest(existing=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                occupied = root / name
                occupied.write_bytes(b"immutable evidence\\n")
                with patch.object(self.generator, "generate", side_effect=AssertionError("Reject before authoring")):
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        self.build(root / "natural.jsonl", root / "manifest.json")
                self.assertEqual(occupied.read_bytes(), b"immutable evidence\\n")
                self.assertEqual(set(root.iterdir()), {occupied})
        for output, manifest in ((FIXED_TOKENIZER, self.root / "manifest.json"),
                                 (self.root / "natural.jsonl", FIXED_TOKENIZER),
                                 (self.root / "same", self.root / "same")):
            with self.subTest(output=output, manifest=manifest), self.assertRaisesRegex(ValueError, "distinct"):
                self.build(output, manifest)
        destination = self.root / "race.jsonl"

        def compete(_source, target):
            Path(target).write_bytes(b"competing evidence\\n")
            raise FileExistsError("another writer won")

        with patch.object(self.generator.os, "link", side_effect=compete):
            with self.assertRaisesRegex(ValueError, "Another writer"):
                self.generator.exclusive_write(destination, b"replacement\\n")
        self.assertEqual(destination.read_bytes(), b"competing evidence\\n")
        self.assertEqual(FIXED_TOKENIZER.read_bytes(), self.tokenizer_raw)
        self.assertEqual(set(self.root.iterdir()), {destination})

    def test_conflicts_and_overlong_complete_dialogues_fail_before_publication(self):
        initial = copy.deepcopy(self.candidates[0])
        other_fold = copy.deepcopy(initial)
        other_fold.update(id="fictional-other-id", groupId="fictional-other-family", split="validation")
        contradiction = copy.deepcopy(initial)
        contradiction["id"] += "-contradiction"
        contradiction["messages"][-1]["content"] = "Un dato contradictorio."
        for injected in (other_fold, contradiction):
            with self.subTest(record=injected["id"]), \
                    patch.object(self.generator, "generate", return_value=[initial, injected]), \
                    patch.object(self.generator, "dialogue_tokens", side_effect=AssertionError("Audit before context")):
                with self.assertRaises(ValueError):
                    self.build()
            self.assertEqual(list(self.root.iterdir()), [])
        with patch.object(self.generator, "dialogue_tokens", return_value=([0] * 258, None)):
            with self.assertRaises(ValueError):
                self.build()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_independent_historical_collision_fixture_is_read_only_and_blocks_publication(self):
        # This is a tiny fictional test fixture created here, never an existing
        # historical or acceptance corpus.  It deliberately occupies a prompt
        # in another fold to exercise the optional authoring collision audit.
        occupied = copy.deepcopy(self.candidates[0])
        occupied.update(id="independent-history", groupId="independent-history-world", split="test")
        occupied["source"] = {"dataset": "independent-fictional-fixture", "kind": "authored-synthetic"}
        path = self.root / "fictional-history.jsonl"
        before = (canonical(occupied) + "\n").encode("utf-8")
        path.write_bytes(before)
        with self.assertRaises(ValueError):
            self.build(historical_corpus_path=path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(FIXED_TOKENIZER.read_bytes(), self.tokenizer_raw)
        self.assertEqual(set(self.root.iterdir()), {path})


if __name__ == "__main__":
    unittest.main()
