# Unidad

Unidad es un proyecto de IA propia que se entrena y ejecuta localmente. El
repositorio reúne modelos, datos, entrenamiento, evaluación y un núcleo común
para conectarlos a distintas aplicaciones mediante Python, consola o API HTTP.
La inferencia utiliza nuestros pesos; no delega las respuestas en otra IA.

**Tenemos una plataforma de desarrollo funcional y modelos experimentales
entrenados. La conversación y el trabajo autónomo fiables siguen pendientes.**
Podemos entrenar, recuperar una corrida, generar respuestas y ejecutar herramientas
con permisos y resultados verificables. Guardar memoria, aprobar pruebas de
software o acumular actualizaciones no demuestra que el modelo haya aprendido a
conversar bien.

[Qué tenemos y cómo encaja cada pieza](docs/QUE_TENEMOS.md) explica el proyecto
con palabras sencillas: los modelos propios, Brain, GNOSIS, NERVA, memoria,
herramientas, datos, checkpoints y lo que falta demostrar.

## Avance comprobado del 13 de octubre de 2026

El [recibo de esta actualización](docs/learning-repair-20261013/progress-20261013-1304.md)
conserva la hora exacta de observación, procesos, fuentes, límites y hashes.
El código y la entrega de pesos del [PR11](https://github.com/guadalupejuan001-code/UNIDAD/pull/11)
se fusionaron en `a948c0fd358b250c7137978e3077ac70cdc51d83`, con los cuatro
checks de CI aprobados. Después se verificaron 45 archivos obtenidos de
`origin/main`, incluidos cinco pesos BEST completos. Esa comprobación valida
los archivos entregados; la capacidad del modelo se mide por sus respuestas.

La base de 15.179.520 parámetros terminó 8.914 actualizaciones y su primer ajuste
conversacional, 8.192. Sus continuaciones propias de 15.769.344 parámetros y
contexto 2.048 completaron entrenamiento de acciones, dos etapas de español
LATAM y retención. Los brazos de copia V2/V1 también terminaron 256
actualizaciones cada uno. Se conservan las referencias, pesos descartados y
[conversaciones anteriores](docs/learning-repair-20261013/comparison/README.md).

Desde la entrega anterior, Tatoeba completó **2.048 actualizaciones causales y
1.024 de retención**. El colector incorporó otro lote de Wikipedia:
**2.651 registros nuevos de 2.648 artículos**, más 2.299 anclas conocidas que
se cuentan como repaso. El lote procede del snapshot de noviembre de 2023;
son datos nuevos para esta continuación, no noticias actualizadas a 2026.
Su etapa causal cerró en **1.772 actualizaciones** por tiempo, seguida de
**1.024 de ajuste**. Son pasos ejecutados, no cantidades de hechos aprendidos.

El supervisor permanece disponible y espera el siguiente lote y su intervalo
de colección. Cada etapa tiene límites finitos de tiempo, RAM y disco; el
entorno debe seguir disponible. Los checkpoints de estas etapas posteriores
están guardados localmente: esta actualización documenta su existencia y sus
hashes, sin presentarlos como nuevos pesos ya publicados en GitHub.

La [última evaluación revisada, posterior a Tatoeba](docs/learning-repair-20261013/comparison/development-tatoeba-retention1024-v1/README.md)
obtuvo **0/40 respuestas completas y 0/10 diálogos** en ambos juicios de IA.
Registraron nueve y diez fallos críticos, respectivamente, y cinco técnicos.
El [candidato anterior a Tatoeba](docs/learning-repair-20261013/comparison/development-wide-latam-retention1024-v1/README.md)
había obtenido 1/40 completa, seis críticos y dos técnicos. Conservamos el
retroceso y las discrepancias; más entrenamiento no garantiza una mejora.
La generación posterior al lote nuevo de Wikipedia aún no tiene revisión
semántica en este corte. Ninguno de estos resultados acredita aceptación.

En autonomía, el modelo consiguió una inspección real en una tarea expuesta y
falló después. Tratamiento y control completaron **0/2 tareas nuevas** cada uno.
El controlador sí comprobó permisos, resultados, conservación de originales y
parada; sus demostraciones con fixtures se identifican aparte. El criterio
conversacional sigue siendo 36/40 respuestas, ocho de diez diálogos y cero fallos
críticos o técnicos en una prueba nueva tras fijar candidato y generación.
La reserva final permanece cerrada y C7 conserva su papel de referencia propia.

## Qué existe y funciona hoy

| Componente | Capacidad implementada | Alcance actual |
| --- | --- | --- |
| Cerebro compartido | `Brain` y `ConversationEngine` sirven los mismos modelos desde Python, CLI y HTTP. | Cada producto configura herramientas y memoria; el modelo no recibe acceso automático a cuentas. |
| Modelos propios | Pesos, configuración, tokenizador y procedencia de Transformers entrenados desde pesos aleatorios. | Son modelos experimentales; C7 se conserva como referencia. |
| GNOSIS | Una configuración reúne cerebro, NERVA, perfiles, recursos y etapas `foundation → conversation → adaptation`. | Las etapas requieren datos y ejecución explícitos; no deciden solas qué aprender. |
| NERVA 0.2 | Backend del modelo pequeño de caracteres, con generación greedy y temperatura cero. | Usa Node.js; C7 y los Transformers siguen ejecutándose mediante PyTorch. |
| Entrenamiento | Preentrenamiento de texto, ajuste conversacional, continuación desde pesos propios y reanudación con optimizador. | Base y primer SFT terminados; acciones y contrastes conservados. Tatoeba cerró 2.048 + 1.024; el primer lote automático de Wikipedia, 1.772 + 1.024. Las etapas posteriores tienen checkpoints locales; no están aceptadas por conversación. GNOSIS conserva su configuración independiente sin corpus. |
| Supervisor local | Adopta una corrida existente, mantiene su presupuesto y recibe nuevas etapas con datos y procedencia fijados. | En el último corte espera datos tras el primer lote automático de Wikipedia. Conserva límites, procedencia y estados; depende del entorno activo y no promueve candidatos automáticamente. |
| Memoria local | Registro de encargos, intentos y correcciones, con búsqueda y referencias al texto original. | Guardar memoria no cambia pesos ni incorpora recuerdos automáticamente al diálogo. |
| Herramientas y autonomía operativa | Controlador experimental recibe propuestas estructuradas del modelo propio, comprueba permisos, ejecuta herramientas registradas y observa/verifica resultados. | Incluye límites, progreso persistente, recuperación, cancelación y aprobaciones para efectos sensibles. Su implementación no demuestra planificación aprendida fiable. |
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
| [`conversation-15m`](models/learning-repair-20261013/conversation-15m/README.md) | 15.179.520 | SFT terminado en 8.192 updates; BEST 4.096. Última respuesta/EOS supervisados, todos los parámetros entrenables. BEST obtuvo 0/40 completas y 0/10 diálogos en DEV; rechazado. |
| `action2048` | 15.769.344 | SFT propio terminado en 256 updates desde extensión de contexto propia, todos los parámetros entrenables. NLL de acciones 0,038118744; planificación/trabajo y conversación de esta etapa aún sin aceptar. |
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

## Entrenamiento terminado y siguiente corrección

La nueva base se entrenó con 10.735 documentos TRAIN y 1.351 VALIDATION de
Wikipedia en español y OpenAssistant. Wikipedia procede del snapshot de noviembre
de 2023; no se presenta como información actualizada a 2026. El tokenizador se
ajustó sólo con TRAIN. Se conservaron familias y particiones históricas, se
eliminaron duplicados y conflictos entre particiones, y TEST quedó fuera del
entrenador y de la caché.

La base cerró por `total_time_budget` en 8.914 updates. Su umbral era 7.200
segundos; el contador final registra 7.337,688, incluida validación y guardado.
Hubo reanudación real con Adam, scheduler, orden y RNG conservados. La NLL causal
pasó de 7,7110 a 2,9819 sobre las mismas 687.867 posiciones de validación.
Los [pesos finales](models/learning-repair-20261013/foundation-15m/README.md),
[BEST/LAST](checkpoints/learning-repair-20261013/foundation-15m/final/README.md)
y la [contabilidad](docs/learning-repair-20261013/foundation-15m-final-target-exposure.json)
conservan ese cierre. Predecir mejor texto no acredita conversación competente.

El SFT posterior entrenó todos los parámetros con `assistant_last_reply`: sólo
la última respuesta y EOS son objetivos. El corpus mezclaba diálogo humano
público y ficción sintética identificada; el 41,295 % de TRAIN era diálogo
humano. Se excluyeron objetivos erróneos y duplicados, conservando historia de
corrección. Terminó en el límite configurado de 8.192 actualizaciones; el runner
registra `absolute_step_boundary` y el estado nativo `stopped`, `reason=signal`.
El tiempo final acumulado es **5.941,433658156999 segundos**. Sus 65.524 ventanas
consumidas contienen 5.286.178 etiquetas supervisadas, incluido EOS. BEST 4.096
se conserva byte por byte y LAST 8.192 tiene un paquete propio. Los
[recibos finales](docs/learning-repair-20261013/conversation-15m-final-delivery.md)
y la [exposición exacta](docs/learning-repair-20261013/conversation-15m-final-target-exposure.json)
separan selección por NLL, actualizaciones y posiciones procesadas.

La [comparación nueva](docs/learning-repair-20261013/comparison/README.md) conserva
80 turnos realmente generados, con la historia propia de cada modelo. Ambos
revisores de IA obtienen 0 completas, 1 parcial y 39 fallidas por modelo; ningún
diálogo completo. C7 tiene 6 críticas/0 técnicas; SFT tiene 12 críticas/6 técnicas.
Cuatro turnos BEST SFT agotan tokens y tres descartan contexto, con uno común. Las
primeras desviaciones ya ocurren con historial íntegro. La NLL de SFT no
compensa este resultado negativo ni permite reemplazar la referencia C7.
La [evaluación real de LAST 8.192](docs/learning-repair-20261013/comparison/development-sft-last8192-v1/README.md)
tampoco acepta la conversación: ambos revisores obtienen 0 completas, 3 parciales,
37 fallidas y 0/10 diálogos. La primaria marca 11 críticas y la cruzada 12;
ambas marcan 8 técnicas. Se conserva el desacuerdo sobre `tap_flow_correction`.
LAST no reemplaza a BEST como selección por validación ni promueve un candidato.
El [tratamiento natural + LATAM, paso 1.024](docs/learning-repair-20261013/comparison/development-natural-latam-step1024-v1/README.md),
también generó 40 turnos reales: ambas revisiones dan 0 completas, 0 parciales,
40 fallidas y 0/10 diálogos; primaria 1 crítica y cruzada 0, ambas con 11 técnicas.
Su única discrepancia crítica interpreta «Tus entradas en dos entradas.».
El tratamiento terminó y tampoco acredita mejora conversacional. Su
[LAST final ya generó los 40 turnos](docs/learning-repair-20261013/comparison/development-natural-latam-last4096-v1/README.md);
la nueva comparación local conserva ambas revisiones, con 0 completas, una
parcial, 14 críticas y una técnica. La intervención regional tampoco comprobó
mejora de conversación, y los datos léxicos no se presentan como conocimiento
aprendido por disponer de pesos o reducir una NLL.

El [cierre del tratamiento](docs/learning-repair-20261013/natural-latam-v4-treatment-final-delivery.md)
conserva 4.096 updates, 32.765 ventanas y 4.277.235 posiciones supervisadas,
incluidas 52.188 posiciones regionales en 1.295 exposiciones regionales. Terminó
por el límite configurado, con razón nativa `signal` y 3.337,367299398 segundos.
BEST 1.024 conserva pesos/configuración/tokenizador de la entrega intermedia;
los metadatos de selección final se guardan aparte. LAST 4.096 conserva su
propio paquete nativo y artefacto de inferencia, más una generación DEV real
con dos revisiones semánticas terminadas; sigue sin aceptación.

El [cierre del control](docs/learning-repair-20261013/natural-latam-v4-control-final-delivery.md)
conserva 4.402.874 etiquetas supervisadas, incluido EOS, y 32.765 ventanas.
BEST 0 reutiliza el padre SFT propio `072eedd2…`; no representa aprendizaje del
control. LAST tiene sus pesos propios nuevos, conservados por separado. En el
[probe regional](docs/learning-repair-20261013/comparison/latam-knowledge-development-probe-v1/README.md),
los cuatro modelos fallan los doce significados TRAIN y seis familias. El autor
del probe también hizo la revisión primaria; la cruzada es de otro agente de
IA, y ninguna es evaluación humana independiente. Los 60 textos, fuentes,
congelación y ambos juicios están íntegros, sin mejora léxica verificada.

La [comparación conversacional del control LAST 4.096](docs/learning-repair-20261013/comparison/development-natural-control-last4096-v1/README.md)
ya conserva ambos juicios: 0 completas, 1 parcial, 39 fallidas, 0/10 diálogos,
2 críticas y 7 técnicas. Coinciden los IDs críticos y las cuatro claves
booleanas por turno; sigue rechazado, sin mejora conversacional verificada.

La siguiente hipótesis reduce el peso de ficción procedural repetitiva y añade
significados regionales revisados. El [currículo natural + LATAM v4](examples/learning-repair-20261013/natural-latam-sft-v4/README.md)
conserva todas las filas humanas/curadas y 1.598 filas procedurales seleccionadas,
frente a las 10.235 anteriores. Se ejecutan **LATAM primero y control después**,
ambos desde el mismo BEST propio 4.096, con configuración igual y hasta 4.096
actualizaciones por brazo. Control contiene 9.951 TRAIN/1.704 VALIDATION;
tratamiento 10.359 TRAIN/1.759 VALIDATION, incluidos 408 TRAIN y 55 VALIDATION
regionales. Las NLL globales de brazos con VALIDATION distintas no se comparan
como si fueran la misma prueba.

La [auditoría con el tokenizador y encoder nativos](docs/learning-repair-20261013/natural-latam-v4-train-audit/README.md)
comprobó sólo TRAIN: retención de cada objetivo/EOS una vez, historia anterior
sin supervisión y ningún prefijo o mensaje actual perdido. Preparar el corpus y
encolarlo no acredita aprendizaje. Se reparó la admisión por campos obligatorios
de procedencia y TEST, reemplazando sólo el supervisor inactivo. En esta lectura
el control terminó en **step=4096**, con 3.321,110810 segundos nativos
acumulados. Acciones también llegó a 256. LANG1 y LANG2 cerraron en 2.048
updates cada uno; la retención, en 1.024. LANG1 procesó 1.445.480 posiciones
supervisadas en 4.096 ventanas y dejó 6.110 ventanas TRAIN sin consumir.
Su umbral cooperativo de 1.800 segundos se sobrepasó en 131,19 segundos;
el motivo nativo conserva `signal`, el del runner `absolute_step_boundary`.
V2 tratamiento cerró en 256, 157,475977 segundos; control V1 también en 256,
131,484277. La lectura histórica de las 11:41:18 UTC registró Tatoeba
recién iniciado en paso 0. El cierre posterior de 2.048 + 1.024 y el siguiente
lote de Wikipedia constan en el recibo actual enlazado al principio. El estado
se puede consultar dentro del entorno:

```sh
python3 examples/continual_training.py status \
  --config examples/learning-repair-20261013/configs/continual-live-collector-v2.json
```

El [bucle progresivo](docs/learning-repair-20261013/loop-runtime.md) conserva etapas
finitas, cola, estados y la activación real del collector V2 sin interrumpir el
entrenador. Espera cuando faltan datos o recursos. Su ejecución depende de este
entorno. Los mensajes personales del creador no forman parte del corpus y los
presupuestos históricos 24+2 horas siguen cerrados.

## Español de Latinoamérica preparado

Las [fuentes regionales de Wikcionario](examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/README.md)
conservan procedencia, revisión, atribución, licencia y particiones. El inventario
directo reúne **22.555 pertenencias a categorías de 19 países/territorios y
11.241 entradas únicas**. La muestra de definiciones incluye 120 entradas:
106 TRAIN/VALIDATION con **229 acepciones**, y 14 TEST privadas excluidas. El
catálogo causal contiene 10.130 filas TRAIN/VALIDATION de metadatos. Ese catálogo
no aporta todos los significados de sus títulos ni representa todo el español.

El SFT regional v4 añade **463 ejemplos sintéticos del proyecto** derivados de
acepciones y notas, no respuestas producidas por Unidad. La revisión rechazó
v2/v3 y retiró los últimos targets dudosos en v4; los originales y sus exclusiones
se conservan. Se corrigen país, interlocutor y formato y se pide aclaración ante
ambigüedad. Las categorías de página no se convierten en significados por país.

La [mezcla causal amplia + LATAM v2](examples/learning-repair-20261013/wide-latam-v2/README.md)
se conserva preparada sin una corrida propia. V2 retiene 10.129 filas del catálogo
tras excluir una familia conflictiva, además de prosa y definiciones regionales.
La [mezcla causal v3 corregida](docs/learning-repair-20261013/wide-latam-v3-enqueue.json)
ya cerró ambos shards en 2.048 updates cada uno y retención en 1.024.
La primera fuente conserva SHA `81c41448…` y [entrega verificable](docs/learning-repair-20261013/wide-latam-v3-language-1-final-delivery.md).
[Tatoeba posterior](docs/learning-repair-20261013/tatoeba-next-v1-enqueue.json)
terminó su etapa causal en 2.048 actualizaciones y la retención en 1.024;
los recibos actuales distinguen esa ejecución de la preparación anterior.
Descargar o mezclar datos no significa que los pesos los aprendieron;
las anclas de retención repetidas no se cuentan como diversidad nueva.

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

Para cargar exactamente el BEST final del tratamiento evaluado y rechazado:

```sh
python -m ia_propia.conversation converse \
  --model models/learning-repair-20261013/natural-latam-v4/intermediate-step001024 \
  --max-new-tokens 160 --temperature 0 --seed 42
```

Su identidad es
`unidad-conversation-ec9ab262aaaa5761dfe77e51232ca9fcf78bea93b7af97ec17d971557f0166df`.
Los [recibos de pesos](docs/learning-repair-20261013/natural-latam-v4-treatment-final-delivery.json)
y las [salidas evaluadas](docs/learning-repair-20261013/comparison/development-natural-latam-step1024-v1/outputs.json)
fijan configuración, tokenizador y SHA de pesos. Cargar este modelo no demuestra
calidad conversacional. La carpeta `final-treatment-best-step001024` contiene
la referencia y metadatos finales; el comando carga el artefacto completo
`intermediate-step001024`, con los mismos pesos finales seleccionados.

El chat GNOSIS carga C7. `plan` calcula recursos sin crear pesos ni iniciar entrenamiento;
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

## Autonomía operativa experimental

El cerebro incorpora `autonomy.start/status/step/run/cancel/approve/resume` por
la misma API autenticada. Un proponente conectado al modelo conversacional
propio genera acciones estructuradas; el controlador valida herramientas,
argumentos y permisos, ejecuta, observa y verifica resultados. Conserva progreso,
intentos y cancelación con límites finitos y recuperación. Los efectos sensibles
requieren aprobación con una credencial de operador separada. Cada producto fija
herramientas y directorios al arrancar; el contexto no puede ampliar ese acceso.

La [demostración ejecutada con acciones de prueba declaradas](docs/learning-repair-20261013/autonomy-demonstration/fixture-v1/evidence.json) leyó archivos reales,
guardó un informe, verificó hashes de originales y canceló otra ejecución antes
de guardar. Eso demuestra el controlador, **no planificación aprendida**. Los
[intentos reales del modelo propio BEST 4.096](docs/learning-repair-20261013/autonomy-demonstration/own-model-v3/evidence.json)
fallaron: la tarea con tres herramientas excedía el contexto (732/512 tokens)
y se bloqueó antes de generar; la tarea mínima sí cabía (336 tokens), pero
produjo texto sin una acción JSON válida y terminó `blocked_model`, sin ejecutar
herramientas ni crear el informe del modelo. No hubo respuestas de respaldo.
Registrar experiencias no reentrena el modelo. La autonomía fiable y la
autoprogramación siguen pendientes.

La verificación de cierre pasó **145 pruebas en 6,893 segundos**, con una CPU
y límite de 180 segundos. Incluye controlador, API, archivos, cerebro anterior,
bucle, collector V2, extensión de contexto y datos de acciones. Comprueba el
software y sus límites; los resultados reales del modelo siguen rechazados.
El CI inicial `38375885022` falló; los [arreglos](docs/learning-repair-20261013/pr10-ci-corrections.json)
conservan las comprobaciones de seguridad y payloads completos. Las ejecuciones
remotas `38379096070` y `38379113739` aprobaron los cuatro checks. El
[recibo](docs/learning-repair-20261013/pr10-publication-verification.json) conserva
la comprobación de CI y de publicación; no convierte los modelos en aceptados.
El [log final de GitHub](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38379113739)
registra 603 pruebas de conversación aprobadas en 75,599 s y 12 ejemplos
aprobados en 7,312 s. La job de Python mínimo informa 521 pruebas en 8,807 s,
con 166 omisiones de dependencias opcionales y resultado OK.

Ya se [exportó un artefacto propio con contexto 2.048](docs/learning-repair-20261013/context2048-extension-delivery.json),
desde el tratamiento 1.024. Conserva byte por byte todos los parámetros heredados
y las primeras 512 posiciones; añade 589.824 parámetros de posición **sin entrenar**,
para un total de 15.769.344. El [ensayo real ampliado](docs/learning-repair-20261013/autonomy-demonstration/context2048-baseline-v1/evidence.json)
ya admite las tres herramientas en 732 tokens, pero el modelo produjo JSON
inválido y terminó `blocked_model`, sin informe ni respuestas de respaldo.
El [corpus de acciones](examples/learning-repair-20261013/autonomy-action-sft-v1/README.md)
conserva 72 tareas ficticias y 216 acciones ejecutadas sobre archivos reales,
192 TRAIN/24 VALIDATION, con etiquetas de fixture declarada. Su
[recibo de encolado SFT](docs/learning-repair-20261013/autonomy-action-v1-enqueue.json)
conserva el orden tras ambos brazos y 256 updates/600 segundos. Ya terminó 256 en
187,049678 segundos; generación y evaluación del candidato propio se registran
aparte. La [demostración posterior](docs/learning-repair-20261013/autonomy-demonstration/action256-own-v1/README.md)
produjo JSON válido en una propuesta, pero pidió `k.txt` sin autorización y
falló; la tarea mínima siguió siendo JSON inválido. El
[reintento opcional del host](docs/learning-repair-20261013/operational-autonomy.md)
permite una nueva propuesta tras argumentos de lectura rechazados, por defecto
cero; su software está probado y la [prueba real con un reintento](docs/learning-repair-20261013/autonomy-demonstration/action256-own-read-repair-v1/README.md)
también falló después de una segunda propuesta mal formada, sin efectos ni informe.

El adaptador de acciones tenía un defecto previo a la generación: texto de
archivos o tareas con marcadores literales como `<system>` bloqueaba el encoder.
La [corrección de serialización](docs/learning-repair-20261013/operational-autonomy.md)
escapa `<`/`>` como `\u003c`/`\u003e` dentro del JSON; `json.loads` recupera
exactamente los datos originales. No omite contenido ni sustituye el modelo.
Tres regresiones específicas elevan la verificación pertinente a 87 pruebas;
no constituyen una nueva demostración de trabajo aprendido.

Para reproducir la demostración con el mismo BEST propio en un directorio nuevo:

```sh
python3 examples/demo_operational_autonomy.py \
  --output-dir private/autonomy-demo-new \
  --model-dir models/learning-repair-20261013/conversation-15m/intermediate-step004096
```

El resultado distingue la ejecución de la fixture, cancelación y el intento real
del modelo; una fixture no sustituye una propuesta fallida del modelo. Para
conectar el controlador desde CLI/HTTP se configuran `--conversation-model`,
`--autonomy-input-root` y `--autonomy-report-root` en `unidad brain call/serve`.
Las dos raíces deben existir y estar separadas. `--autonomy-max-steps` y
`--autonomy-max-seconds` fijan los límites. El [estado actual](docs/learning-repair-20261013/current-progress.md)
separa funcionamiento técnico, conversación y planificación.

La [comparación de datos de copia](docs/learning-repair-20261013/autonomy-action-copy-v2-enqueue.json)
ya ejecutó el tratamiento V2: 256 updates, BEST 256 NLL 0,322699097, modelo
propio `e70ea28e…`; control V1 cerró también256. Ambos parten
del mismo `e202eafc…`, con iguales 256 updates/600 segundos. Las
[instrucciones de comparación](docs/learning-repair-20261013/operational-autonomy.md)
exigen respuestas y efectos reales con la misma política; bajar NLL no acepta
argumentos, trabajo ni conversación. El padre fijado de Tatoeba se conserva.

El [contraste operativo real V2/V1](docs/learning-repair-20261013/autonomy-action-copy-v2-real-matched-development.json)
conserva los tres primeros prompts, archivos, herramientas y generación
idénticos. Padre E202 pidió `k.txt` y control 734c pidió `v.txt`, ambos bloqueados
antes del handler. Tratamiento E70 pidió `dos.txt`, lectura realmente verificada,
y falló después por JSON inválido. Ninguno creó su informe; originales iguales,
sin sustitución de respuestas ni planificación aceptada. Es una sola tarea DEV
expuesta, insuficiente para generalización; las NLL de corpus distintos no se comparan.

La [comparación con dos tareas nuevas](docs/learning-repair-20261013/action-copy-v2-development-tasks-v1/summary.json)
congeló nombres, objetivos y ambos candidatos antes de generar. Tratamiento y
control completaron **0/2 tareas**, sin ejecutar herramientas ni crear informes;
los originales permanecieron intactos. Los argumentos inventados y el JSON
inválido siguen bloqueando la autonomía aprendida. Estos casos pasan a desarrollo
y quedan excluidos del entrenamiento futuro; no acreditan la aceptación conversacional.

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
La adaptación de parámetros y la autoprogramación siguen pendientes de
demostraciones específicas. El controlador operativo implementado conserva
experiencias y resultados reales; guardarlos no reentrena el modelo ni acredita
que haya aprendido a planificar.

## Integración de las ramas

Los cambios compatibles y la evidencia de PR1–10 se reúnen en `main`, sin
entregas en borrador. Se conservan las ramas originales como historia.

| Origen | Resultado integrado |
| --- | --- |
| [PR1](https://github.com/guadalupejuan001-code/UNIDAD/pull/1) y [PR3](https://github.com/guadalupejuan001-code/UNIDAD/pull/3) | Evaluador común y documentos originales de diagnóstico/redacción. Los candidatos rechazados conservan ese estado. |
| [PR2](https://github.com/guadalupejuan001-code/UNIDAD/pull/2) | Proyección de vocabulario en la última posición al generar, conservando las funciones posteriores de GNOSIS. |
| [PR4](https://github.com/guadalupejuan001-code/UNIDAD/pull/4) | Exportador ONNX y pruebas; publicación sin sobrescribir y control de cambios del modelo. |
| [PR5](https://github.com/guadalupejuan001-code/UNIDAD/pull/5) y [PR6](https://github.com/guadalupejuan001-code/UNIDAD/pull/6) | GNOSIS, NERVA, entrenamiento reanudable y cargador mmap. |
| [PR7](https://github.com/guadalupejuan001-code/UNIDAD/pull/7) | Continuación propia de español, supervisor, datos/configuraciones distribuibles y recibos intermedios. |
| [PR8](https://github.com/guadalupejuan001-code/UNIDAD/pull/8) | Reparación del registro de transición del supervisor, pesos y BEST/LAST finales de la base, fuentes públicas nuevas y mezcla de español en cola. El SFT estaba ejecutándose en la observación de PR8; ahora su primera etapa está cerrada. Integrar archivos no acepta sus respuestas. |
| [PR9](https://github.com/guadalupejuan001-code/UNIDAD/pull/9) | SFT intermedio y fuentes/colas posteriores, conservados sin aceptar la conversación. La entrega posterior de PR10 añade cierres, evaluaciones y autonomía experimental; ya está integrada en main con CI aprobado. |
| [PR10](https://github.com/guadalupejuan001-code/UNIDAD/pull/10) | Fusionado sin borrador en `62cf4d4`, con cuatro checks CI aprobados. Conserva las correcciones, cierres, datos, pesos propios y autonomía experimental; conversación y planificación aprendida siguen sin aceptar. |
| [PR11](https://github.com/guadalupejuan001-code/UNIDAD/pull/11) | Fusionado sin borrador en `a948c0f`, con cuatro checks CI aprobados. Añade reintentos limitados de lectura, observaciones JSON sin pérdida, entrenamiento propio, checkpoints recuperables y resultados reales, incluidos fallos en tareas nuevas. |
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

- [Qué tenemos en Unidad](docs/QUE_TENEMOS.md) y [último avance comprobado](docs/learning-repair-20261013/progress-20261013-1304.md)
- [GNOSIS: configuración, comandos y límites](docs/gnosis.md)
- [NERVA dentro del cerebro](docs/nerva.md)
- [Entrenamiento conversacional](docs/conversation.md) y [base lingüística](docs/language-pretraining.md)
- [Fuentes y licencias de esta continuación](examples/learning-repair-20261013/DATASOURCES.md) y [ampliación preparada](examples/learning-repair-20261013/wide-spanish-v1/README.md)
- [Memoria y encargos](docs/workspace.md) y [núcleo para productos](docs/brain.md)
- [Estado de esta entrega](docs/learning-repair-20261013/current-progress.md), [comparación real nueva](docs/learning-repair-20261013/comparison/README.md) y [cierre SFT](docs/learning-repair-20261013/conversation-15m-final-delivery.md)
- [Estado y fuentes anterior](docs/ESTADO_ACTUAL.md), [observación histórica 06:44](docs/learning-repair-20261013/observation-20261013-064409-sft-active-pr8-merged.json) y [estado estructurado anterior](docs/unidad-current-status.json)
- [Órdenes del creador](ORDENES_USUARIO.md) e [instrucciones de desarrollo](AGENTS.md)
- [Antecedentes y comandos del README anterior](docs/historial-readme-20261013.md)

Los pesos y recibos de experimentos descartados se conservan como evidencia.
Publicarlos no los convierte en modelos aceptados. Los seis ajustes anteriores
terminaron y conservan su estado; la base de 15,18 M y su primer SFT también
terminaron. Tratamiento y control regionales terminaron en 4.096 cada uno;
acciones terminó en 256; v3 cerró ambos shards en 2.048 y retención en 1.024.
Copia V2/V1 cerró ambos brazos en 256; Tatoeba, en 2.048 + 1.024, y
el primer lote automático de Wikipedia, en 1.772 + 1.024. Los presupuestos
históricos 24+2 horas siguen cerrados. Clonar esta documentación no inicia una corrida ni un supervisor.
