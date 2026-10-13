# Unidad

IA privada de uso personal. Núcleo independiente para conectarlo a productos e
interfaces distintos, conservar contexto y avanzar hacia un trabajador digital.
Unidad funciona localmente con Python. El núcleo usa NumPy; el nuevo modelo
conversacional usa PyTorch y un tokenizador propio. No requiere Hormiga-X,
una API de IA ni pesos preentrenados.

La [configuración GNOSIS](configs/gnosis.json) reúne cerebro, NERVA y etapas de
aprendizaje. Incluye admisión de recursos, entrenamiento por microbatches y
soporte explícito de arquitecturas mayores. Se comprobó cálculo y manejo de
artefactos de 48,85 millones de parámetros aleatorios; esto no demuestra una
mejora conversacional. [Uso, mediciones y límites](docs/gnosis.md).

Las [órdenes del creador](ORDENES_USUARIO.md) están adjuntas al proyecto. Definen
el objetivo de Unidad y distinguen lo realizado de lo pendiente. Las instrucciones
para quienes modifiquen el código están en [AGENTS.md](AGENTS.md).

**Estado verificado el 12 de octubre de 2026: la conversación competente sigue
pendiente; no hay entrenamiento activo comprobado en este entorno.** C7 se conserva
como referencia experimental: **7 completas, 5 parciales, 28 fallidas, seis críticas
y 0/10 diálogos completos** en los 40 turnos conocidos, según la revisión primaria
de IA del proyecto. Los candidatos posteriores de la comparación relacional
fallaron la conservación de capacidades y no se promovieron. Los presupuestos
anteriores de 24 horas y dos horas están cerrados.

El [estado actual y sus fuentes](docs/ESTADO_ACTUAL.md) distingue resultados,
versiones, ramas y límites. Los pesos de C7 están incluidos en `main`; el
[paquete completo de pesos y checkpoints](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012)
está publicado en la rama de artefactos. Esa publicación conserva experimentos;
no declara aceptado ningún modelo. C7 tiene **6.983.937 parámetros**, incluidos
los de copia v1; la base anterior sin esa cabeza tiene 6.967.296.

**Núcleo conectable con modelos experimentales.**
Ya puede ejecutar operaciones y flujos definidos por el producto, usando las
herramientas que ese producto le proporciona. Los modelos actuales todavía no
comprenden encargos abiertos ni planifican el trabajo de forma competente.
Una red pequeña aprende una tarea concreta; una inteligencia que resuelva encargos
abiertos necesita capacidades y pruebas adicionales. No hemos replicado a una
persona ni su forma completa de pensar.

Los antecedentes siguientes conservan los resultados de sus etapas; no describen
procesos activos ni presupuestos disponibles actualmente.

El sucesor propio sin cabeza de copia tiene **6.967.296 parámetros**, BPE de 8.192
tokens y contexto de 512 tokens. Completó 6.000 pasos de preentrenamiento desde
pesos aleatorios y 6.000 de ajuste conversacional, pero falló la comprobación
con 40 turnos nuevos: **2 respuestas completas, 3 parciales y 35 fallidas;
0 de 10 diálogos completos**. No se aceptó ni promovió como modelo
predeterminado. El [informe de v7](docs/unidad-conversation-language-v7-results.json)
conserva la evidencia; el objetivo de conversación básica sigue pendiente.

Una comparación posterior completó **384 pasos por brazo** para comprobar un
objetivo adicional de ranking entre respuestas y contextos. El control obtuvo
0 completas, 2 parciales y 38 fallidas en la revisión primaria; la cruzada
registró 1 completa, 1 parcial y 38 fallidas. Con ranking, ambas registraron
0 completas, 1 parcial y 39 fallidas. Ningún brazo logró un diálogo completamente
correcto ni superó la promoción. Son 40 turnos de desarrollo ya expuestos, revisados por
asistentes de IA del proyecto. El [informe del A/B](docs/unidad-counterfactual-ranking-v1-results.json)
conserva la validación y las diferencias; el ranking no se adoptó. La etapa
posterior usó muestreo uniforme sin reemplazo y pérdida media por token desde
los pesos propios de v7. Se guardó a los 2.786 pasos para cambiar la precisión
del entrenamiento, conservando el checkpoint seleccionado por validación del
paso 2.000. En 40 turnos de desarrollo, la revisión primaria obtuvo **2 completas,
6 parciales y 32 fallidas**; la cruzada, **2 completas, 4 parciales y 34 fallidas**.
Ninguna obtuvo un diálogo completo. El [informe de esa etapa](docs/unidad-uniform-coverage-b1-results.json)
conserva los desacuerdos y el rechazo. Las etapas posteriores con BF16 también
terminaron; sus resultados y cierres están identificados en el estado actual.

El entrenador ahora permite activar `train_precision="cpu_bfloat16"`. En dos
comparaciones cortas con orden invertido, el tiempo mediano por paso pasó de
**0,914 a 0,501 s**. Una prueba nativa pequeña verificó reanudación exacta y
conservación del modo predeterminado. El [informe de precisión](docs/unidad-cpu-bfloat16-v1.json)
y la [guía](docs/conversation.md#entrenar-con-bf16-en-cpu) explican cómo usarlo;
el efecto sobre la conversación se evalúa aparte.

La memoria local ahora reutiliza la reconstrucción validada del registro y
permite buscar encargos con referencias al texto original y la corrección
vigente. En una medición caliente con 2.001 eventos ficticios, consultar el estado
pasó de **20,5 a 0,7 ms** y buscar de **50,3 a 7,8 ms**. Son tiempos de esta CPU,
con carga de fondo; la búsqueda usa el mismo algoritmo léxico en ambos caminos.
El [informe de memoria](docs/unidad-memory-optimization-v1.json) y la
[guía reproducible](docs/workspace.md#medir-la-memoria-local) conservan alcance,
pruebas y límites. Estas operaciones no entrenan pesos ni añaden recuerdos
automáticamente a las conversaciones.

## Qué contiene

- Un especialista de atención, entrenado desde pesos aleatorios y separado de
  Hormiga-X con su evidencia original. Predice una métrica numérica, no creatividad.
- Un modelo de lenguaje pequeño con entrenamiento propio. Su demostración sirve
  para comprobar que cambian los pesos y se puede generar texto; no demuestra
  comprensión ni calidad conversacional.
- Transformers conversacionales propios desde pesos aleatorios: se conservan
  las etapas pequeñas de 1.088.256 parámetros y el sucesor de 6.967.296.
  Aprenden español con textos públicos y ejemplos ficticios identificados.
  Permiten entrenamiento en segundo plano, checkpoints y reanudación; su
  calidad debe comprobarse por separado de las pruebas del código.
- Un registro local de encargos, intentos y correcciones en texto libre. No impone
  categorías de personalidad. Permite preparar ejemplos aprobados para entrenar,
  sin confundir memoria con aprendizaje de parámetros.
- Un núcleo Python común para CLI y HTTP, con los pesos incluidos en el paquete.
  Cada «cuerpo» conecta sus herramientas y su memoria; varios cuerpos pueden
  compartir memoria si su configuración lo solicita expresamente.
- Ejecución de flujos explícitos: por ejemplo, generar con un modelo y pasar el
  resultado a una herramienta del producto. Esto comprueba la conexión y la
  ejecución; la calidad del texto sigue siendo la del modelo experimental.

## Instalar

Python 3.11 o posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
unidad --help
python -m unittest discover -s tests -v
```

El paquete también se puede usar con `python -m ia_propia` después de instalarlo.
`ia_propia` es el nombre interno del paquete; el comando público es `unidad`.
El comando anterior `ia-propia` se conserva como alias. Cambiar el nombre del
proyecto conserva los pesos y las identidades de los modelos ya entrenados.
Los datos personales van en `private/`, excluido de Git. Los artefactos de `models/`
y textos de `examples/` son demostraciones identificadas, no datos personales.
Los modelos que entrenes con tus propios textos también deben guardarse en
`local-models/` o `private/`: pueden memorizar fragmentos del corpus. Si eliges
otra ruta, la exclusión de Git puede dejar de aplicar. Los permisos locales
no son cifrado.

## Primeros pasos

La referencia conversacional propia conservada es [Unidad
C7](models/conversation-reference-c7/README.md). Sigue siendo experimental: obtuvo 7
respuestas completas de 40 y ningún diálogo completamente correcto de 10. Sus
[conversaciones reales](models/conversation-reference-c7/evidence/dialogues-actual.txt),
pesos y comando comprobado están guardados; todavía no cumple la aceptación de
conversación básica.

Para el nuevo modelo de conversación, instala las dependencias CPU:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[conversation]'
```

Para abrir C7 con el motor y las condiciones comprobadas:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse \
  --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

La prueba de consola conservó el historial real y reprodujo cuatro respuestas
evaluadas. Una consola funcional no acredita conversación competente.

La [guía de entrenamiento conversacional](docs/conversation.md) explica cómo
preparar datos, empezar, comprobar el progreso, guardar un modelo y reanudar.
Las versiones siguientes son antecedentes conservados. Hay una
[instantánea experimental guardada](models/conversation-basic-v1/README.md)
con pesos, configuración y tokenizador:

```bash
unidad conversation chat --model models/conversation-basic-v1 \
  --message "Hola, ¿cómo empezamos?" --max-new-tokens 64 --seed 42
unidad conversation status --run-dir private/conversation/runs/basic-v1
```

La consulta de estado corresponde al entorno donde se inició el trabajo; una
clonación nueva tiene los pesos guardados, pero no un entrenador en ejecución.
El primer corpus básico combina 120 ejemplos inventados con 1.087 respuestas
públicas revisadas de OASST1: 965 registros de entrenamiento, 115 de validación
y 127 reservados para prueba, separados por familia de conversación. Los textos
de este chat y las órdenes del creador no se utilizaron como corpus personal.

El estilo solicitado —confidente, cercano y que conecte puntos— orienta los
ejemplos y el contexto del modelo; eso todavía no demuestra que lo haya aprendido.
La [evidencia de esta etapa](docs/unidad-conversation-validation-v0.3.0.json)
registra pesos, métricas y respuestas reales, sin reemplazar su texto por respuestas
de otra IA. El presupuesto original de entrenamiento fue de hasta 24 horas;
permitía parar antes por validación o cierre del entorno y está cerrado.
Unidad 0.3.0 pasó 139 pruebas con las dependencias conversacionales. También
se comprobó una instalación del paquete fuera del repositorio, con Python y HTTP
usando el mismo modelo entre dos cuerpos. Las respuestas de la instantánea
todavía contienen errores y repeticiones: estas pruebas acreditan el código
y la integración, no un habla competente.

La versión 0.4 añade continuación entre etapas con `--initial-model`. La primera
corrida se detuvo al estancarse; la segunda amplió el corpus a 14.207 registros
y conservó sus mejores pesos y el mismo tokenizador. La ampliación mantuvo
las familias reservadas y usó el presupuesto restante de las 24 horas originales.
El [registro de progresión](docs/unidad-conversation-progression-v0.4.0.json)
conserva la comparación y el estado observado. Esta etapa sigue siendo experimental.
Su [instantánea exportada](models/conversation-spanish-v2/README.md) se puede
usar desde los productos sin instalar el corpus. Unidad 0.4 pasó 152 pruebas
conversacionales y técnicas; sus respuestas todavía requieren mejoras.
La segunda etapa fue preparada y arrancada mediante comandos explícitos.
Unidad todavía no decide sola qué aprender ni modifica su código.

```bash
unidad conversation chat --model models/conversation-spanish-v2 \
  --message "Hola, ¿cómo empezamos?" --max-new-tokens 64 --seed 42
unidad conversation status --run-dir private/conversation/runs/expanded-v1
```

La versión 0.5 conecta el interior del proyecto mediante un contrato común de
mensajes, límites y parámetros, y un `ConversationEngine` compartido por la consola,
el cerebro y la evaluación. El entrenamiento exporta los mismos pesos, configuración
y tokenizador que consume ese motor. Se conserva un único comportamiento de
inferencia en todos esos caminos.

La etapa histórica siguiente priorizó diálogos cortos con contexto completo.
Las ventanas deslizantes de entrenamiento conservan texto anterior y enseñan cada token de
la respuesta final una sola vez. La batería de desarrollo contiene 28 casos
en 14 pares que cambian la petición actual, con candidatos intercambiados;
también conserva las respuestas reales para revisión humana:

```bash
unidad conversation assess --model models/conversation-spanish-v2 \
  --cases examples/conversation-evaluation.json \
  --output private/conversation/development-report.json
```

Este comando no entrena, no evalúa el test reservado y no finaliza una corrida.
Las probabilidades de candidatos son un diagnóstico: deben revisarse junto con
lo que el modelo realmente dice. Las respuestas actuales siguen siendo
experimentales; conectar el código no acredita comprensión ni programación
autónoma. Pasaron 174 pruebas y se verificó el mismo resultado desde consola,
Python y HTTP en una instalación externa. El
[registro de esta integración](docs/unidad-conversation-cohesion-v0.5.0.json)
conserva pruebas, datos, procedencia y las respuestas defectuosas observadas.

La etapa concentrada se pausó cuando empezó a sobreajustarse. Se conserva su
[mejor instantánea experimental](models/conversation-context-v3/README.md),
y la etapa posterior combinó 5.608 diálogos breves con un ritmo de aprendizaje
menor. Usó el tiempo restante del mismo presupuesto original, ahora cerrado. La
[guía de conversación](docs/conversation.md) explica ambos recorridos.

```bash
unidad conversation chat --model models/conversation-context-v3 \
  --message "Hola." --max-new-tokens 64 --seed 42
unidad conversation status --run-dir private/conversation/runs/balanced-short-v1
```

También se conserva una [instantánea de bases conversacionales](models/conversation-basics-v4/README.md)
con respuestas y errores documentados. Sigue siendo experimental: los saludos
han mejorado, pero la conversación libre todavía confunde identidades y datos.
Puedes probarla en una sesión de terminal con historial temporal:

```bash
unidad conversation converse --model models/conversation-basics-v4 \
  --temperature 0 --max-new-tokens 32
```

Se adaptó el aprendizaje para evitar calcular salidas sin supervisión. En la
CPU disponible, la [comparación medida](docs/unidad-learning-optimization-v1.json)
redujo un 13,5 % el tiempo por paso y conservó la inferencia. La etapa posterior
redujo repeticiones de ejemplos fáciles y añadió diversidad de textos y
correcciones. El [registro de aprendizaje](docs/unidad-learning-curriculum-v2.json)
conserva la comparación de respuestas y la nueva etapa: las correcciones y los
hechos de conversación libre siguen fallando. Pasaron 277 pruebas técnicas;
mejorar su conversación debe comprobarse por separado. La
[guía](docs/conversation.md) explica cómo activar el nuevo modo de entrenamiento.

La versión 0.5.2 añade una política opcional de práctica adaptativa: reutiliza
las pérdidas de `train` para dar más prioridad a lo difícil, con repaso y límites
de prioridad. Se actualiza por bloques cortos y guarda ese estado para reanudar
exactamente. Los casos de validación y test no ajustan esas prioridades. La
[guía de configuración](docs/conversation.md#adaptar-la-práctica-a-lo-que-todavía-cuesta)
explica la mezcla. En la [comparación controlada](docs/unidad-adaptive-learning-v0.5.2.json),
la configuración adaptativa conservó mejor algunos cambios de nombre, pero
empeoró la validación y la copia literal. Sigue desactivada por defecto; se
conservan los pesos anteriores y no se afirma que esta política haya acortado
el tiempo necesario para aprender a conversar.

La versión 0.6.0 añade una etapa previa de lenguaje: `objective="causal_lm"`
aprende los tokens de documentos de español con procedencia y particiones
propias. Después puede ajustarse el mismo modelo con conversaciones, conservando
sus pesos y tokenizador propios. La [guía de preentrenamiento](docs/language-pretraining.md)
explica los formatos, la reanudación y una configuración nativa de 6.967.296
parámetros con contexto de 512 tokens. La [CI de la optimización de memoria](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38168377960)
pasó **320 pruebas con dependencias conversacionales, 12 de ejemplos y 254 en
la instalación mínima**, en el commit `ee3d5c2`.
Son comprobaciones técnicas separadas de la aceptación del modelo; disponer de
esta ruta de aprendizaje no demuestra que el nuevo modelo converse bien.

La [base pública de español](docs/unidad-spanish-language-base-v1.json) conserva
la procedencia y las particiones de 3,58 millones de palabras de entrenamiento,
equivalentes a 6,29 millones de objetivos con el BPE propio de 8.192 tokens.
Combina textos de Wikipedia con mensajes humanos revisados de OASST; el ajuste
posterior usa diálogos humanos y ejemplos ficticios identificados del proyecto.
El [preparador reproducible](examples/prepare_public_spanish_language.py) recibe
las rutas y hashes de los corpus revisados y conserva familias completas. No
lee el historial personal ni descarga pesos de otros modelos.

Se conservan la [base española de 6.000 pasos](models/spanish-foundation-v1/README.md)
y el [candidato SFT v7 de otros 6.000 pasos](models/conversation-language-v7/README.md).
El SFT comenzó desde los pesos propios de la base y conservó su configuración
y tokenizador, con optimizador y programador nuevos. Se seleccionó el paso
6.000 por validación:
NLL de texto 4,77545 en preentrenamiento y NLL 1,63739 con reducción por secuencia
en SFT. Esas pérdidas tienen objetivos y reducciones distintos; no equivalen a
recuentos de tokens ni demuestran conversación competente.

En la evaluación nueva de v7, la revisión primaria y la cruzada, realizadas por
asistentes de IA del proyecto, coincidieron en 2 respuestas completas, 3
parciales y 35 fallidas, con 0 de 10 diálogos completos. Los 40 turnos se
generaron sin errores ni truncamiento. No se superó el criterio de 36 respuestas
completas, 8 diálogos completos y cero fallos críticos. Se mantiene explícita
la diferencia original de 13 frente a 12 fallos críticos entre las revisiones;
tras compararlas, ambos asistentes confirmaron una unión de 14 turnos críticos,
sin cambiar las 40 calificaciones ni el rechazo. No se consultó TEST ni se
promovió v7 como modelo
predeterminado. El [informe público](docs/unidad-conversation-language-v7-results.json)
incluye procedencia, hashes y límites; no es una evaluación externa.

También se conserva el [checkpoint uniforme B1](models/conversation-language-uniform-b1/README.md).
Para probar sus pesos experimentales en una conversación local:

```bash
unidad conversation converse --model models/conversation-language-uniform-b1 \
  --temperature 0 --max-new-tokens 32
```

La [instantánea v5](models/conversation-basics-v5/README.md) conserva una mejora
experimental: 13 respuestas completas de 22 turnos de desarrollo, frente a 6
del padre. Aún falla al recuperar ciudades y puede repetir sílabas. Para probarla:

```bash
unidad conversation converse --model models/conversation-basics-v5 \
  --temperature 0 --max-new-tokens 32
```

La [instantánea experimental v6](models/conversation-basics-v6/README.md)
conserva otra etapa con más repaso de español público y actualizaciones más
pequeñas. Obtuvo 15 respuestas completas de los mismos 22 turnos de desarrollo;
siguen fallando las ciudades, algunos títulos y una pregunta sobre un dato
desconocido. La etapa terminó y conserva el paso 600 seleccionado por validación.
El [informe de refinamiento](docs/unidad-learning-refinement-v0.5.2.json)
registra el cambio conjunto de currículo y tasa de aprendizaje. Son casos de
desarrollo conocidos, no una prueba nueva de conversación general. Para probar
este candidato, usa `--model models/conversation-basics-v6` en el comando anterior.
Sus 15 respuestas completas de 22 turnos conocidos no son comparables
directamente con los 40 turnos nuevos de v7.

También se compararon señales directas de aprendizaje de atención y selección
de datos del diálogo. La segunda prueba elevó la copia literal de 37 a 50 casos
exactos de 89 frente al control experimental ya entrenado, sin repetirlo.
Quedó por debajo de v6 en conversación libre, por lo que se conserva como
investigación y no modifica el modelo usable. El [diagnóstico de aprendizaje](docs/unidad-copy-learning-diagnostics-v1.json)
registra las dos pruebas, sus resultados y los cambios descartados.

El núcleo funciona también desde una instalación del paquete fuera del repositorio:

```bash
unidad brain capabilities --body mi-producto --state-dir private/brain
```

Para integrarlo en Python, servir su API HTTP o conectar herramientas, sigue
[la guía del cerebro y sus cuerpos](docs/brain.md). El ejemplo
[dos productos](examples/brain_in_two_products.py) usa el mismo núcleo con dos
conjuntos de herramientas. Ninguno llama a otra IA.

Unidad 0.2.0 pasó 86 pruebas, incluida la ejecución de un modelo hacia una
herramienta y el aislamiento entre productos. También se comprobó el paquete
instalado fuera del repositorio. La evidencia está en
[el registro de validación del núcleo](docs/unidad-brain-validation-v0.2.0.json).

El registro de encargos también conserva su interfaz anterior:

```bash
unidad init --workspace private/default
unidad task add --workspace private/default \
  --goal "Describe aquí un trabajo concreto" \
  --context "Qué importa hoy, fuentes disponibles y resultado deseado"
unidad task list --workspace private/default
```

El especialista ya guardado se puede reproducir sin entrenarlo de nuevo:

```bash
unidad trend predict --model models/attention-v1.json \
  --input examples/attention-observations.json \
  --as-of 2024-10-16T00:00:00Z --historical
```

Con las 30 capturas iniciales, su error en test todavía supera el de conservar
el último valor. El resultado está conservado en el informe original.

También hay un modelo de lenguaje ya entrenado con los diez textos inventados
de demostración, con 2.903 parámetros. Puedes probar su generación:

```bash
unidad language generate --model models/language-demo.json \
  --prompt "Una tarea " --length 160 --seed 42 --temperature 0.8
```

Su texto todavía contiene palabras sin sentido. Los pesos y resultados
comprueban un entrenamiento pequeño; no acreditan capacidad para resolver
encargos. La verificación inicial está en
[el registro de Unidad 0.1.0](docs/unidad-validation-v0.1.0.json).
El [registro inicial](docs/validation-v0.1.0.json) conserva la primera verificación,
realizada antes de elegir el nombre.

La evaluación del lenguaje conserva sólo el estado recurrente necesario cuando
no calcula gradientes, reduciendo memoria temporal. `unidad task list` devuelve
resúmenes; `unidad task show` permite consultar un historial completo. Las
[mediciones de memoria](docs/language-optimization.md) conservan el resultado
y sus límites; los pesos entrenados no cambiaron.

El registro guarda encargos; guardarlos no inicia un flujo. Los intentos pueden ser
trabajos tuyos o de un asistente que tú revisas. Una corrección actualiza el
contexto registrado; para cambiar pesos hay que exportar ejemplos aprobados y
entrenar expresamente. No usamos este chat como corpus automáticamente.

Los comandos y límites están explicados en [el espacio de trabajo](docs/workspace.md),
[el modelo de lenguaje](docs/language.md) y [el especialista de atención](docs/attention.md).
La separación de Hormiga-X se documenta en [procedencia](docs/provenance.md).

```mermaid
flowchart LR
    A[Encargo y contexto libres] --> B[Intento de trabajo]
    B --> C[Tu corrección o aprobación]
    C --> D[Registro local revisable]
    D --> E[Exportación explícita de ejemplos aprobados]
    E --> F[Entrenamiento y evaluación]
    F --> G[Pesos de un modelo propio]
```

El núcleo puede ejecutar una secuencia de operaciones suministrada por el
producto. Convertir un encargo libre en un plan útil y comprobar su calidad
sigue pendiente; se evaluará sobre encargos reales.

## Hacia un trabajador digital

La mejora del modelo se comprobará sobre encargos reales del usuario:
qué resultado necesita, qué información puede usar y cómo sabremos que terminó.
A partir de ahí se podrán evaluar las herramientas conectadas y tareas completas.
La capacidad de ejecutar acciones y la calidad del modelo deben demostrarse por
separado; generar texto no basta para acreditar que hizo el trabajo.

Las conexiones ejecutan únicamente herramientas registradas por quien integra
el producto. No se obtiene acceso a cuentas al guardar un contexto. El
entrenamiento y las exportaciones se solicitan con comandos explícitos.
