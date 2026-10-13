# Historia ideal y generada: experimento offline v1

Estos dos corpus contienen los mismos 2112 registros: 1056 TRAIN y 1056
VALIDATION, con las mismas 88 familias, preguntas y respuestas finales. Los
diálogos son ficticios y proceden del currículo sintético del proyecto, revisado
por asistentes de IA. No son conversaciones privadas del creador ni una
evaluación humana independiente.

| Archivo | Contenido | SHA256 |
| --- | --- | --- |
| `ideal.train-validation.jsonl` | Historia original revisada | `a0137e595b7e84f6b6e87d4d1baad0e346c3ba2ed4b5ab1da81eef236df5b8b8` |
| `generated.train-validation.jsonl` | Historia anterior generada por C7 en TRAIN | `7ea1ebf926ebd810d622c53a7041684310a3b3b019e75e22b8cf1c5eacd214ff` |
| `teacher-replies-actual.jsonl` | 576 respuestas reales del teacher y sus entradas ficticias | `aa01b98a4e1247e89a4a491714b49266d4793d74b0e2b6f0076205a10c2be86b` |

En el brazo generado cambian únicamente respuestas assistant anteriores de
368 registros TRAIN. Las preguntas y la respuesta final supervisada permanecen
idénticas; VALIDATION también es idéntico. Los anclajes de identidad/estilo
conservan su historia. Las salidas del teacher quedan completas en el archivo
raw, incluidos sus errores.

El teacher fue el modelo propio C7, con linaje desde pesos aleatorios, no un
modelo externo: ID
`unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2`,
pesos SHA256 `666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836`.
La generación offline usó temperatura 0, semilla 42 y hasta 32 tokens nuevos.
No fue aprendizaje en línea durante el entrenamiento.

Los registros mantienen su `kind=authored-synthetic` y los campos heredados,
incluido `modelResponsesUsed=false`; esas etiquetas describen la fuente de los
objetivos originales. La historia anterior del brazo generado sí contiene
salidas reales de C7, documentadas por `historyPolicy`, el manifest y el raw del
teacher. Los campos se copiaron sin cambiar etiquetas.

## Procedencia, licencia y reproducción

Se incluyen `manifest.json`, `design.json` y `progress.json` originales. El
manifest identifica el currículo, revisión, teacher, tokenizador y código por
SHA256. La fuente ya está en
[`conversation-grounded-curriculum-v2.jsonl`](../conversation-grounded-curriculum-v2.jsonl)
y su [revisión de alcance](../conversation-grounded-curriculum-v2.review.json).
Los textos y hechos son sintéticos del proyecto; los registros no declaran una
licencia de datos independiente. No se les atribuye la licencia Apache-2.0 de
OASST ni autoría humana. El repositorio no declara una licencia global adicional.

Para regenerar el par con el teacher exacto, desde la raíz del repositorio:

```bash
python examples/prepare_on_policy_history.py \
  --corpus examples/conversation-grounded-curriculum-v2.jsonl \
  --teacher-model models/conversation-reference-c7 \
  --output /tmp/unidad-on-policy-history-v1 \
  --families-per-skill 4 \
  --generation-budget-seconds 600
```

La salida debe ser nueva. La regeneración requiere el checkpoint propio y
código compatible con los hashes del recibo; cambios de generación pueden
producir historias diferentes. Para repetir los entrenamientos originales,
consume directamente estos corpus fijados y verifica sus hashes. Este par no
contiene TEST y su preparación no utilizó casos reservados de aceptación.

La revisión heredada de plantillas no certifica cada respuesta ni demuestra
conversación competente. Los recibos originales se conservan también en
[docs/data](../../docs/conversation-integration-20261013/data/on-policy-history-v1/).
