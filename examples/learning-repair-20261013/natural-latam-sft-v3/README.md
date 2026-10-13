# Currículo natural y español de Latinoamérica v3

**Conservado como evidencia; no admitir v3 como corpus positivo.** La revisión
final encontró defectos también en las acepciones alternativas de caraota y
algarrobilla. La [versión final v4](../natural-latam-sft-v4/README.md) retira sus
seis filas y conserva todos los demás registros regionales byte por byte.

Datos preparados para comparar dos ajustes desde el **mismo checkpoint propio**,
con 4.096 actualizaciones, configuración y condiciones de generación comunes.
El entrenamiento actual conserva su corpus. Estos archivos describen una etapa
posterior; por sí solos no acreditan aprendizaje ni conversación competente.

La hipótesis es reducir la frecuencia excesiva de ficción sobre objetos y
ubicaciones, observada como respuestas de ubicación ante otras peticiones. Se
retienen 400 familias completas: diez por cada una de las 40 plantillas TRAIN,
con sus prefijos originales disponibles, 1.598 filas. Todos los ejemplos
humanos y curados originales y sus targets se mantienen byte por byte; también
las 1.704 filas VALIDATION anteriores. No se reparte de nuevo ninguna familia.

| Brazo | TRAIN | VALIDATION | Ficción procedural en TRAIN |
| --- | ---: | ---: | ---: |
| Control natural | 9.951 | 1.704 | 1.598; 16,06 % |
| Natural + LATAM | 10.365 | 1.759 | 1.598; 15,42 % |

Cada brazo conserva 7.676 filas humanas y 677 curadas originales en TRAIN.
LATAM añade 414 TRAIN y 55 VALIDATION: 469 registros, 91 familias léxicas y 126
acepciones distintas seleccionadas de Wikcionario. Los diálogos mencionan los
19 países/territorios registrados en las mediciones. Esto es una muestra; no
representa un catálogo completo ni todas las variedades de cada país.

Los targets regionales son sintéticos del proyecto, generados de forma
determinista desde acepciones documentadas; no son respuestas de Unidad ni
revisión humana. Incluyen significado por país, formato breve, una pregunta de
aclaración sólo ante ambigüedad real, país corregido, varios interlocutores y
recuperación de una respuesta sobre la tarea equivocada. Hay ocho patrones de
conducta: variar nombres o preguntas no se cuenta como conocimiento nuevo.
Un error anterior del asistente sólo aparece como contexto; nunca se publica
su prefijo como objetivo positivo. La política sigue siendo
`assistant_last_reply`.

La revisión previa rechazó v2. V3 excluye nueve acepciones con defectos y las
abreviaturas de uso aún sin resolver para el lector; conserva literalmente
fuente, versiones anteriores y [motivos](target-exclusions.json). No inventa
correcciones biológicas. Países y calificadores proceden de cada acepción;
categorías de página y etiquetas genéricas como «América» no se expanden a
países. Las notas de uso, incluidas las despectivas o históricas, se conservan
en los sentidos supervisados. La familia malformada de «chompa» no se reasigna:
queda excluida sólo del derivado hasta corregir su extracción en otra versión.

La preparación real consumió 0,629 segundos y alcanzó 191,92 MiB de RSS, con
un proceso, sin importar modelos/tokenizadores ni leer TEST. Las entradas
regionales tienen mediana de 38 palabras y máximo de 104; su longitud con el
tokenizador propio se comprueba antes de entrenar. Nueve pruebas pertinentes
pasaron, incluidas seis referencias de país/acepción para guagua, chamba y china.
Son comprobaciones de datos, no pruebas de capacidad aprendida. La revisión
semántica completa del proyecto se conserva en `qa` cuando queda concluida.

Los dos corpus tienen VALIDATION diferentes. Sus NLL globales no se comparan
como si midieran lo mismo: la comparación conversacional usa los mismos 40
turnos de desarrollo, con historia realmente generada, el mismo motor y los
mismos límites. Sólo un candidato/configuración fijados pasan a una prueba
final nueva. No hay promoción automática por pérdida o existencia de pesos.

Para reproducir los datos en una carpeta nueva:

```bash
python examples/prepare_natural_latam_sft.py \
  --prior examples/learning-repair-20261013/sft/sft.train-validation.jsonl \
  --regional examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/definitions/latin-american-wiktionary.train-validation.jsonl \
  --design examples/learning-repair-20261013/natural-latam-sft-v3/prospective-design.json \
  --target-exclusion-receipt examples/learning-repair-20261013/natural-latam-sft-v3/target-exclusions.json \
  --output-dir private/reproduction-natural-latam-v3
```

Los hashes de corpus y de contenido conversacional, sin metadatos, están en
[natural-control.manifest.json](natural-control.manifest.json) y
[natural-latam.manifest.json](natural-latam.manifest.json). Los bytes del corpus
son reproducibles; tiempo y RSS de la ejecución varían. Se conservan el helper
congelado y la [receta](recipe.json).

Las adaptaciones regionales conservan CC BY-SA 4.0 y atribuciones, revisiones,
historiales y URL de licencia por registro. El corpus combinado es una colección
con las condiciones originales por componente: OpenAssistant conserva Apache
2.0; los datos anteriores conservan su procedencia/licencias declaradas en
[SFT original](../LICENSE-DATA.md). No se relicencian esos componentes por agruparlos.
