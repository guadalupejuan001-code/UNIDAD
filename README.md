# Unidad

**Proyecto de IA propia, local y experimental, con modelos entrenados desde cero
y un cerebro compartido para distintas aplicaciones.**

Unidad reúne pesos propios, entrenamiento, evaluación, memoria persistente y
herramientas configuradas por la aplicación. La visión es desarrollar una IA
personal que converse en español, realice trabajo útil y pueda adaptarse mediante
aprendizaje verificable. La inferencia actual funciona localmente con Python,
NumPy y PyTorch, sin llamar a una API de otra IA.

**Estado observado el 13 de octubre de 2026, 06:44 UTC:** la base propia de
**15.179.520 parámetros** terminó con **8.914 actualizaciones** y **22.906.669
posiciones objetivo procesadas**. El ajuste conversacional desde esa base ya
está entrenando: **2.200 actualizaciones** en la lectura, con todos los parámetros.
Hay otra ampliación pública de español en cola. **La conversación fiable sigue
pendiente**: todavía no se generaron ni evaluaron diálogos de esta nueva etapa.
El [recibo actual](docs/learning-repair-20261013/observation-20261013-064409-sft-active-pr8-merged.json)
se añade a las observaciones históricas sin sobrescribirlas.

El código compatible de PR1–6 está integrado en `main`: evaluador, generación,
ONNX, GNOSIS, NERVA y datos en disco. [PR7](https://github.com/guadalupejuan001-code/UNIDAD/pull/7)
también está fusionado y sin borrador: incorpora esta continuación y el
supervisor. [PR8](https://github.com/guadalupejuan001-code/UNIDAD/pull/8),
fusionado sin borrador, integra la reparación de transición, los artefactos
finales de la base y la ampliación pública de español. `main` está en
`34e5fd0c772305acdb18616c684c7c4cf7450cac` en esta observación.
Los seis ajustes anteriores completaron
2.560 actualizaciones y ninguno pasó la aceptación; sus
[resultados](docs/conversation-integration-20261013/README.md) se conservan.
La [continuación de aprendizaje](docs/learning-repair-20261013/current-delivery.md)
explica los datos nuevos, límites y estado comprobado de la corrida activa.

## Qué existe y funciona hoy

| Componente | Capacidad implementada | Alcance actual |
| --- | --- | --- |
| Cerebro compartido | `Brain` y `ConversationEngine` sirven los mismos modelos desde Python, CLI y HTTP. | Cada producto configura herramientas y memoria; el modelo no recibe acceso automático a cuentas. |
| Modelos propios | Pesos, configuración, tokenizador y procedencia de Transformers entrenados desde pesos aleatorios. | Son modelos experimentales; C7 se conserva como referencia. |
| GNOSIS | Una configuración reúne cerebro, NERVA, perfiles, recursos y etapas `foundation → conversation → adaptation`. | Las etapas requieren datos y ejecución explícitos; no deciden solas qué aprender. |
| NERVA 0.2 | Backend del modelo pequeño de caracteres, con generación greedy y temperatura cero. | Usa Node.js; C7 y los Transformers siguen ejecutándose mediante PyTorch. |
| Entrenamiento | Preentrenamiento de texto, ajuste conversacional, continuación desde pesos propios y reanudación con optimizador. | Base propia de 15,18 M terminada y SFT activo con corpus fijado; GNOSIS conserva su configuración independiente sin corpus. |
| Supervisor local | Adopta una corrida existente, mantiene su presupuesto y recibe nuevas etapas con datos y procedencia fijados. | Proceso real observado vivo; espera nuevos datos cuando falta una etapa. No garantiza aprendizaje continuo fuera del entorno ni promoción de candidatos. |
| Memoria local | Registro de encargos, intentos y correcciones, con búsqueda y referencias al texto original. | Guardar memoria no cambia pesos ni incorpora recuerdos automáticamente al diálogo. |
| Herramientas y flujos | Ejecuta operaciones y herramientas registradas mediante pasos explícitos. | Los pasos los aporta la aplicación; todavía no existe un planificador aprendido. |
| Especialistas pequeños | Generación por caracteres y predicción de una métrica de atención. | Son demostraciones con errores y límites documentados, no capacidades generales aceptadas. |

Las rutas privadas se excluyen de Git. La selección de modelos, herramientas,
backends y recursos corresponde al anfitrión; las peticiones no pueden cambiarlos.
La API permite construir otros clientes, incluido Android, pero esta integración
no conecta automáticamente una aplicación Android ni acredita una prueba física.

## Los modelos: entrenamiento y escalado son cosas distintas

| Modelo o perfil | Parámetros | Qué se comprobó |
| --- | ---: | --- |
| [C7](models/conversation-reference-c7/README.md) | 6.983.937 | Referencia propia entrenada, con pesos publicados, vocabulario BPE de 8.192 tokens y contexto de 512. Su conversación no pasa la aceptación. |
| [`foundation-15m`](models/learning-repair-20261013/foundation-15m/README.md) | 15.179.520 | Base causal terminada: 8.914 actualizaciones, BPE de 2.048 tokens y contexto de 512. Pesos y BEST/LAST finales publicados mediante PR8; sin conversación aceptada. |
| `conversation-15m` | 15.179.520 | SFT activo desde esa base propia: 2.200 actualizaciones observadas a las 06:44 UTC. Última respuesta/EOS supervisados, todos los parámetros entrenables; pendiente de respuestas nuevas. |
| `50m` | 48.852.992 | Un modelo real inicializado aleatoriamente calculó gradientes de 64 tokens, se exportó y se cargó con resultados idénticos. Tuvo **cero actualizaciones de pesos**. |
| `100m` | 99.446.400 | Perfil validado y conteo exacto. No se ejecutó ni entrenó aquí. |
| `1b` | 991.586.304 | Perfil validado y estructura meta, sin asignar sus pesos en CPU. No se ejecutó ni entrenó aquí. |

**El modelo de 49 millones no es C7 ampliado ni un modelo que haya aprendido
conversación.** Fue una prueba local de cálculo y manejo de artefactos. En `main`
están el [runner reproducible](examples/verify_gnosis_scaling.py) y su
[evidencia](docs/gnosis-scaling-20261013.json); no se publicó allí un checkpoint
conversacional entrenado de ese tamaño.

El código admite hasta 1.200 millones de parámetros mediante un
`parameter_budget` explícito. Las configuraciones anteriores conservan el límite
de 20 millones. Un perfil no proporciona el hardware, los datos ni las horas de
entrenamiento que necesita. El entrenamiento actual es CPU. El cargador opcional
`data_loading="mmap"` prepara corpus y ventanas en disco para muestreo uniforme;
GPU sigue pendiente. Los [controles y mediciones del cargador](docs/tokenized-data-20261013.md)
no acreditan una mejora de las respuestas.

## La continuación que está entrenando

La nueva base se entrenó con 10.735 documentos TRAIN y 1.351 VALIDATION de Wikipedia
en español y OpenAssistant. Wikipedia procede del snapshot de noviembre de
2023; es texto nuevo para esta corrida, no información actualizada a 2026.
El tokenizador se ajustó sólo con TRAIN. Se conservaron las familias y
particiones históricas de OpenAssistant, se eliminaron duplicados exactos y
familias con texto compartido entre particiones, y TEST quedó fuera del
entrenador y de la caché.

La base cerró por `total_time_budget` en 8.914 updates, antes del máximo de
12.000. El umbral era 7.200 segundos; el contador final registra 7.337,688
segundos al incluir validación y guardado de cierre. La parada es cooperativa,
por lo que el umbral no constituye un límite exacto de tiempo de pared. Hubo
reanudación real con Adam, scheduler, orden y RNG conservados. La NLL causal
validada pasó de 7,7110 a 2,9819 sobre las mismas 687.867 posiciones. Esa métrica
mide predicción de texto; la conversación se comprobará después del ajuste.
Los [pesos finales](models/learning-repair-20261013/foundation-15m/README.md),
[BEST/LAST](checkpoints/learning-repair-20261013/foundation-15m/final/README.md)
y [contabilidad exacta](docs/learning-repair-20261013/foundation-15m-final-target-exposure.json)
conservan la evidencia. Estos archivos se incorporaron a `main` mediante PR8.

Para consultar el estado real en el entorno de entrenamiento:

```sh
python3 examples/continual_training.py status \
  --config private/learning-repair-20261013/configs/continual.json
```

El [supervisor](docs/learning-repair-20261013/continual-loop.md) puede esperar
nuevos corpus indefinidamente mientras él y el entorno sigan vivos. Cada etapa
tiene límites finitos; repetir datos no se presenta como conocimiento nuevo.
El colector y la generación de desarrollo al cerrar SFT están habilitados con
fuentes fijadas, pero aún no se ejecutaron en esta lectura. Los mensajes
personales del creador no forman parte del corpus. C7 mantiene sus pesos y
su condición de referencia hasta comprobar otro candidato con diálogos nuevos.

La [reanudación real](docs/learning-repair-20261013/foundation-15m-first-resume.json)
conserva corpus y código congelados; no reinicia Adam ni el tiempo nativo.
La etapa SFT activa usa todos los parámetros y supervisa sólo la última
respuesta y EOS. El 41,295 % de sus registros TRAIN procede de diálogo humano
público; el resto es ficción sintética identificada. Se excluyeron siete
objetivos erróneos TRAIN y uno VALIDATION, conservando su texto como historia
en prefijos correctivos posteriores, y se retiraron cinco diálogos exactos
repetidos. El ajuste tiene hasta 8.192 updates y umbral de 7.200 segundos;
sus respuestas aún deben evaluarse. En el paso 2.048, la NLL SFT de validación
fue 1,7358, frente a 2,7381 al iniciar este ajuste. Es otro objetivo y otro
conjunto de datos: no se compara directamente con la NLL causal de la base ni
acredita conversación competente. Un
[error real de transición](docs/learning-repair-20261013/continual-stage-transition-repair.json)
bloqueó al supervisor después de cerrar la base. Se corrigió una colisión del
nombre `kind` en el registro de eventos, una razón de estado obsoleta y la
identificación de alias del mismo ejecutable Python. Se reemplazó sólo el
supervisor bloqueado; la base no se reinició y SFT comenzó desde sus pesos.

Los [corpus y configuraciones preparados](examples/learning-repair-20261013/continual-template.json)
permiten reproducir dos etapas finitas en otro clon. La plantilla requiere
rellenar `workspaceRoot`; su colector está deshabilitado y después espera
nuevos datos. La configuración activa de esta sesión es distinta. Los
[checkpoints](checkpoints/learning-repair-20261013/foundation-15m/README.md)
conservan la versión intermedia y los estados finales en partes de hasta
32 MiB, con hashes. Restaurarlos y comprobar su publicación son operaciones
separadas; no acreditan conversación competente.

La [ampliación pública en cola](examples/learning-repair-20261013/wide-spanish-v1/README.md)
añade 2.406 registros de literatura, ensayos/gramática, frases comunitarias y
léxico, con 1.470.331 palabras entre TRAIN y VALIDATION. Conserva 2.299 mensajes
OpenAssistant como retención declarada. El archivo contiene 4.084 TRAIN,
621 VALIDATION y cero TEST. Seguirá al SFT actual con una etapa causal y otro
SFT de retención; todavía no se entrenó. Las fechas y licencias quedan por
fuente: descargar Tatoeba en 2026 no convierte sus frases principalmente de
2007–2012 en información factual actual. No se cuentan las anclas repetidas
como diversidad nueva.

## Qué respondieron los ajustes anteriores

Se entrenaron desde los pesos propios de C7 y V7, con CPU, Adam y todos los
parámetros, tres experimentos acotados: diálogo humano público en español,
historia ideal frente a historia realmente generada, y 120 diálogos ficticios
originales de cuatro turnos con anclas humanas. Los autores de estos 120 diálogos
son agentes de IA del proyecto; no son conversaciones humanas ni respuestas de
Unidad. TRAIN, VALIDATION y TEST conservan familias separadas. TEST no se
introdujo en el entrenador.

| Ajuste | Pasos reales | Snapshot seleccionado por validación | Completas en desarrollo | Diálogos completos | Críticas / técnicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| C7, referencia sin modificar | — | — | 1/40 | 0/10 | 4 / 0 |
| Humano, desde V7 | 256 | 256 | 0/40 | 0/10 | 0 / 1 |
| Humano, desde C7 | 256 | 256 | 1/40 | 0/10 | 0 / 1 |
| Historia ideal, desde C7 | 512 | 512 | 1/40 | 0/10 | 2 / 0 |
| Historia generada, desde C7 | 512 | 384 | 1/40 | 0/10 | 0 / 0 |
| Diálogo natural, desde V7 | 512 | 128 | 0/40 | 0/10 | 0 / 0 |
| Diálogo natural, desde C7 | 512 | 128 | 0/40 | 0/10 | 0 / 1 |

La pérdida de validación bajó, pero no hubo ganancia neta de respuestas completas.
Las críticas inferiores también reflejan omisiones y textos incoherentes; no
acreditan mejores conocimientos. Por eso no se amplió una intervención fallida
ni se reemplazó C7. Estos recuentos de desarrollo son de una batería nueva y
no se suman a las evaluaciones históricas.

El candidato `history-generated`, paso 384, quedó fijado por identidad y hashes
antes de abrir otra prueba reservada. La aceptación exige **36/40 respuestas
completas, 8/10 diálogos correctos y cero fallos críticos o técnicos**. Se usan
40 turnos en diez diálogos, historial realmente generado, 160 tokens por
respuesta, temperatura cero y semilla 42. Las revisiones son de IA del proyecto,
**no evaluaciones humanas independientes**.

En esa prueba reservada nueva, la revisión primaria obtuvo:

| Modelo | Completas | Parciales | Fallidas | Diálogos completos | Críticas / técnicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| C7 conservado | 0/40 | 4 | 36 | 0/10 | 2 / 0 |
| Historia generada, paso 384 | 0/40 | 1 | 39 | 0/10 | 2 / 0 |

Ambos fallaron el criterio. No hubo respuestas vacías, recortes de contexto ni
agotamiento del presupuesto de tokens en esos 80 turnos. La prueba ya consumida
se conserva como evidencia; no se usó para cambiar los pesos ni la selección.
La revisión cruzada también obtuvo cero completas y cero diálogos correctos;
se conservan sus diferencias sobre parciales y críticas.

Los [pesos publicados](models/conversation-integration-20261013/README.md),
[checkpoints reanudables](checkpoints/conversation-integration-20261013/README.md)
y el [informe](docs/conversation-integration-20261013/README.md) permiten
inspeccionar y reproducir el resultado. Guardar un checkpoint no mantiene un
entrenamiento activo. Todavía no hay una comparación que demuestre calidad
competitiva frente a una IA comercial.

## Qué se optimizó de verdad

Se incorporaron microbatches, escritura de artefactos por bloques y carga sin
inicializar otra copia completa del modelo. La reanudación se comprobó en FP32 y
BF16, también después de un fallo en mitad de un lote. La admisión fija corpus y
arquitectura antes de cargar pesos y rechaza rutas de etapas que se solapen.

En la medición CPU del modelo aleatorio de 48,85 millones de parámetros:

| Operación | Pico RSS nuevo | Pico RSS de referencia |
| --- | ---: | ---: |
| Exportación | 651 MB | 1.237 MB |
| Carga | 710 MB | 876 MB |

Son observaciones específicas, no un ahorro universal; el RSS residente final de
la carga no mejoró. En una prueba pequeña separada, los microbatches redujeron un
**87,46 %** las reservas de activaciones retenidas por autograd, conservando la
pérdida con diferencias mínimas de gradientes. Esa cifra no representa la RAM total.

NERVA reprodujo **448 caracteres** de NumPy en siete generaciones; el arranque de
Node añadió tiempo. No se demostró aceleración con ese backend.

La integración final aprobó **494 pruebas y 12 pruebas de ejemplos** con las
dependencias de conversación y ONNX instaladas. El
[recibo de software](docs/conversation-integration-20261013/software/verification.json)
fija código y logs. La proyección final conserva cuatro generaciones C7 y
observó 1,91–2,03 veces de aceleración en el cálculo del siguiente token, bajo
condiciones concretas de CPU. La [paridad ONNX](docs/conversation-integration-20261013/runtime/README.md)
conservó cuatro generaciones greedy; no se probó en un teléfono físico.
GitHub Actions comprueba el código integrado. Estas pruebas acreditan
funcionamiento técnico, no comprensión ni calidad conversacional.
Las [mediciones anteriores](docs/gnosis.md) y el
[cargador mmap](docs/tokenized-data-20261013.md) conservan sus propios límites.

## Probar el proyecto

Python 3.11 o posterior. Desde un clon del repositorio, en Linux/CPU:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[conversation]'
unidad --help
```

Para usar la configuración GNOSIS con su backend NERVA, instala también
**Node.js 18 o posterior**. Después:

```sh
unidad gnosis plan --project configs/gnosis.json
unidad gnosis capabilities --project configs/gnosis.json --body pc
unidad gnosis chat --project configs/gnosis.json \
  --max-new-tokens 160 --temperature 0 --seed 42
```

Para cargar exactamente el candidato experimental evaluado (también rechazado):

```sh
python -m ia_propia.conversation converse \
  --model models/conversation-integration-20261013/history-generated \
  --max-new-tokens 160 --temperature 0 --seed 42
```

Su identidad termina en `c2a558a2`; el [recibo de carga](docs/conversation-integration-20261013/software/published-model-cli.json)
verifica los cuatro hashes y la consola con ese artefacto. Esta comprobación de
carga no demuestra calidad conversacional.

El chat carga C7. `plan` calcula recursos sin crear pesos ni iniciar entrenamiento;
la configuración entregada informa que los corpus están sin configurar. Clonar
pesos y checkpoints no inicia un proceso que siga aprendiendo.

Desde Python, dos productos pueden compartir el mismo cerebro:

```python
from ia_propia.gnosis import load_project

project = load_project("configs/gnosis.json")
pc = project.connect("pc")
android = project.connect("android")
# Mismo cerebro y modelos; memoria separada por producto.
```

La [guía GNOSIS](docs/gnosis.md) explica las etapas, los formatos de datos,
presupuestos, exportación y evaluación. La [guía del cerebro](docs/brain.md)
describe el protocolo HTTP y las herramientas por producto. Los datos personales
y modelos entrenados con ellos deben conservarse en `private/` o `local-models/`.
Las órdenes y los mensajes de este chat no se usan automáticamente como corpus.

## Potencial y prioridades

El potencial de Unidad está en poder controlar sus datos, pesos y evolución:
entrenar una base propia, especializarla en tareas delimitadas y utilizarla desde
varias aplicaciones mediante el mismo núcleo. GNOSIS permite organizar nuevas
etapas sin rehacer cada interfaz y conservar la procedencia de lo aprendido.
La memoria y las herramientas ofrecen una base para construir un asistente
personal útil cuando el modelo pueda comprender y realizar esas tareas de forma fiable.

Ese potencial depende de resolver la calidad del aprendizaje. La prioridad es:

1. Ampliar y diversificar la base lingüística de español, conservando procedencia,
   derechos de uso y separación entre entrenamiento, validación y evaluación.
2. Conseguir conversación que mantenga hechos, siga correcciones y responda a la
   petición actual; medirla con diálogos nuevos e historias realmente generadas.
3. Escalar datos, contexto y modelos con recursos adecuados, utilizando el
   cargador en disco e incorporando GPU.
4. Comprobar adaptación y retención entre etapas antes de promover nuevos pesos.
5. Evaluar después herramientas, tareas completas y propuestas de modificación
   de código, cada una con criterios propios.

Acercarse a asistentes comerciales es un objetivo de investigación, no un
resultado garantizado por usar NERVA, aumentar parámetros o dedicar más horas.
La adaptación automática y la autoprogramación forman parte de la visión futura;
necesitan demostraciones específicas y todavía no están implementadas como
capacidades autónomas del modelo.

## Integración de las ramas

Los cambios compatibles y la evidencia de PR1–8 se reúnen en `main`, sin
entregas en borrador. Se conservan las ramas originales como historia.

| Origen | Resultado integrado |
| --- | --- |
| [PR1](https://github.com/guadalupejuan001-code/UNIDAD/pull/1) y [PR3](https://github.com/guadalupejuan001-code/UNIDAD/pull/3) | Evaluador común y documentos originales de diagnóstico/redacción. Los candidatos rechazados conservan ese estado. |
| [PR2](https://github.com/guadalupejuan001-code/UNIDAD/pull/2) | Proyección de vocabulario en la última posición al generar, conservando las funciones posteriores de GNOSIS. |
| [PR4](https://github.com/guadalupejuan001-code/UNIDAD/pull/4) | Exportador ONNX y pruebas; publicación sin sobrescribir y control de cambios del modelo. |
| [PR5](https://github.com/guadalupejuan001-code/UNIDAD/pull/5) y [PR6](https://github.com/guadalupejuan001-code/UNIDAD/pull/6) | GNOSIS, NERVA, entrenamiento reanudable y cargador mmap. |
| [PR7](https://github.com/guadalupejuan001-code/UNIDAD/pull/7) | Continuación propia de español, supervisor, datos/configuraciones distribuibles y recibos intermedios. |
| [PR8](https://github.com/guadalupejuan001-code/UNIDAD/pull/8) | Reparación del registro de transición del supervisor, pesos y BEST/LAST finales de la base, fuentes públicas nuevas y mezcla de español en cola. El SFT está ejecutándose; integrar archivos no acepta sus respuestas. |
| [Rama de artefactos](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012) | Archivo de paquetes históricos descargables, conservado en su rama. |

La [matriz de reconciliación](docs/conversation-integration-20261013/branch-reconciliation.json)
identifica las cabezas y 324 documentos importados sin cambiar sus bytes.
`source-pointer-v2.patch` permanece como evidencia experimental; no se aplica.
Los textos antiguos describen el estado de su fecha. Integrar su historia no
promueve modelos fallidos ni modifica sus respuestas.

NERVA incluye además un lenguaje e intérprete con operaciones de tensores,
diferenciación automática, actualización de parámetros y checkpoints. Es una
pieza de experimentación que merece describirse por sí misma. Su adaptador actual
dentro de Unidad sólo realiza inferencia del pequeño modelo de caracteres; no
entrena C7 ni ejecuta su Transformer.

## Documentación y antecedentes

- [GNOSIS: configuración, comandos y límites](docs/gnosis.md)
- [NERVA dentro del cerebro](docs/nerva.md)
- [Entrenamiento conversacional](docs/conversation.md) y [base lingüística](docs/language-pretraining.md)
- [Fuentes y licencias de esta continuación](examples/learning-repair-20261013/DATASOURCES.md) y [ampliación preparada](examples/learning-repair-20261013/wide-spanish-v1/README.md)
- [Memoria y encargos](docs/workspace.md) y [núcleo para productos](docs/brain.md)
- [Estado y fuentes](docs/ESTADO_ACTUAL.md), [continuación activa observada](docs/learning-repair-20261013/observation-20261013-064409-sft-active-pr8-merged.json) y [estado estructurado anterior](docs/unidad-current-status.json)
- [Órdenes del creador](ORDENES_USUARIO.md) e [instrucciones de desarrollo](AGENTS.md)
- [Antecedentes y comandos del README anterior](docs/historial-readme-20261013.md)

Los pesos y recibos de experimentos descartados se conservan como evidencia.
Publicarlos no los convierte en modelos aceptados. Los seis ajustes anteriores
terminaron y conservan su estado; la base de 15,18 M terminó y su SFT está activo
en la observación indicada arriba. Los presupuestos históricos 24+2 horas siguen
cerrados. Clonar esta documentación no inicia una corrida ni un supervisor.
