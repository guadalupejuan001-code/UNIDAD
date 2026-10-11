# Conversación en español desde pesos propios

Unidad incorpora un Transformer pequeño para aprender a responder en español.
Sus pesos comienzan aleatorios; el código no descarga pesos preentrenados ni
consulta otra IA para generar las respuestas. Es una primera etapa experimental:
tener un checkpoint o reducir la pérdida no demuestra conversación competente,
comprensión general ni capacidad para trabajar o programarse por sí misma.

La configuración básica tiene **1.088.256 parámetros** cuando el vocabulario BPE
alcanza 2.048 tokens: dimensión 128, cuatro capas, cuatro cabezas y contexto de
256 tokens. Usa atención causal, normalización previa, GELU y una salida que
comparte pesos con los embeddings. El tokenizador se ajusta sólo con `train`.
Si un corpus pequeño produce un vocabulario menor, cambia el número de parámetros;
el artefacto conserva la configuración y el recuento reales.

## Instalar para CPU

Desde la raíz del repositorio, con Python 3.11 o posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[conversation]'
unidad conversation --help
```

Las dependencias opcionales son PyTorch 2.8.0, tokenizers 0.22.2 y safetensors
0.7.0. La instalación CPU evita exigir una GPU. La memoria, el número de hilos y
la velocidad disponibles siguen limitando el entrenamiento. El controlador de
trabajos actual usa mecanismos de Linux para bloqueo y seguimiento de procesos.
El núcleo anterior funciona sin estas dependencias si no se configura el nuevo
modelo conversacional.

## Datos de la primera etapa

[examples/conversation-demo.jsonl](../examples/conversation-demo.jsonl) contiene
120 registros inventados para el proyecto, identificados como
`authored-synthetic`. Son diálogos ficticios de español básico y del estilo
cercano solicitado. Los textos de este chat, el historial personal y
[las órdenes del creador](../ORDENES_USUARIO.md) no se usaron como corpus personal.

La fuente pública es [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
licencia Apache-2.0, revisión
`fdf72ae0827c1cda404aff25b6603abec9e3399b`. El importador conserva procedencia,
identificadores, idioma y revisión, y filtra mensajes según sus etiquetas de
revisión. Importa conversaciones en español; los ejemplos sintéticos externos
quedan excluidos. El archivo original y su licencia se conservan junto a los
datos locales. El importador no descarga el corpus.

El [registro de procedencia y resultados](unidad-conversation-validation-v0.3.0.json)
incluye la URL oficial fija, el SHA-256 del archivo original y los conteos de la
importación. La [licencia Apache-2.0 de esa fuente](licenses/OASST1-Apache-2.0.txt)
se conserva en el repositorio. También hay una
[instantánea experimental ya exportada](../models/conversation-basic-v1/README.md)
que puedes probar sin descargar el corpus ni entrenar de nuevo:

```bash
unidad conversation chat --model models/conversation-basic-v1 \
  --message 'Hola, ¿cómo empezamos?' --max-new-tokens 64 --seed 42
```

Si ya tienes el archivo de árboles OASST de esa revisión en la ruta indicada:

```bash
unidad conversation import-oasst \
  --input private/conversation/sources/oasst-ready.trees.jsonl.gz \
  --output private/conversation/oasst-es.jsonl \
  --language es \
  --revision fdf72ae0827c1cda404aff25b6603abec9e3399b
```

El [preparador básico](../examples/prepare_basic_conversations.py) selecciona
hasta 500 familias públicas de diálogos breves, sin bloques de código, conserva
sus splits y elimina diálogos exactamente duplicados. Para preparar el corpus
que usa la etapa básica:

```bash
python examples/prepare_basic_conversations.py \
  --demo examples/conversation-demo.jsonl \
  --oasst-corpus private/conversation/oasst-es.jsonl \
  --human-groups 500 \
  --output private/conversation/basic-es-v1.jsonl
```

La selección realizada con estas fuentes produjo 1.207 registros: 120 sintéticos
del proyecto y 1.087 públicos, con 965 en entrenamiento, 115 en validación y
127 en test. Son registros de conversación; varios pueden pertenecer a la misma
familia. El helper imprime los conteos y el SHA-256 del archivo realmente
producido. Una fuente o revisión diferente puede cambiar esos conteos.

La separación ocurre por familia completa: una conversación y sus continuaciones
permanecen en un solo split. El corpus canónico es JSONL con `id`, `groupId`,
`split`, `source` y `messages`. La carga rechaza IDs repetidos y familias presentes
en varios splits. La procedencia describe los datos; no es texto que se enseñe
al modelo como respuesta.

El helper básico rechaza un archivo de salida existente. El importador OASST
puede reemplazar su salida; para repetir cualquiera de estos pasos elige un nombre
nuevo y conserva intacto el corpus de una corrida que quieras reanudar.
`private/` y `local-models/` están excluidos de Git.

## Entrenar y seguir el proceso

`train` trabaja en primer plano. Esta prueba corta usa los 120 ejemplos ficticios
y un directorio nuevo:

```bash
unidad conversation train \
  --corpus examples/conversation-demo.jsonl \
  --run-dir private/conversation/runs/demo-v1 \
  --budget-seconds 60
```

`start` crea un trabajador local en segundo plano. Para una nueva corrida básica,
guarda este objeto como
`private/conversation/config/basic-training-v2.json`:

```json
{
  "batch_size": 8,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "grad_clip": 1.0,
  "max_epochs": 5000,
  "max_steps": 400000,
  "max_seconds": 86400,
  "validation_every": 100,
  "checkpoint_every": 100,
  "patience": 20,
  "min_delta": 0.0001,
  "seed": 42,
  "warmup_steps": 100,
  "torch_threads": 4
}
```

```bash
unidad conversation start \
  --corpus private/conversation/basic-es-v1.jsonl \
  --run-dir private/conversation/runs/basic-v2 \
  --training-config private/conversation/config/basic-training-v2.json
unidad conversation status --run-dir private/conversation/runs/basic-v2
```

`basic-v1` es el nombre de la primera corrida iniciada en este entorno; se puede
consultar con `status --run-dir private/conversation/runs/basic-v1`. Los comandos
anteriores usan `basic-v2` para iniciar una corrida nueva. Un directorio existente
no se reinicia ni se sobrescribe mediante `start`.

`max_seconds` es el presupuesto total acumulado de esa corrida, limitado a
86.400 segundos. `--budget-seconds` permite acotar una invocación sin reiniciar
ese total. El proceso también puede terminar por límites de épocas o pasos,
por una parada solicitada o por falta de mejora en validación. Entrenar durante
24 horas no obliga a ignorar una validación que empeora.

El estado JSON informa `status`, `reason`, `step`, `epoch`, `elapsedSeconds`,
`lastCheckpoint`, `bestCheckpoint`, `bestValidationNll`, `processAlive` y
`heartbeatAgeSeconds`, cuando están disponibles. `planned` significa iniciado
pero aún en preparación; `running`, que el entrenamiento publicó actividad;
`stopped`, una pausa; `completed`, un límite alcanzado; y `failed`, un error.
Si el estado registrado era `planned` o `running` y el proceso ya no está vivo,
la consulta informa `stale`. Una corrida `stopped`, `completed` o `failed`
conserva ese estado aunque el proceso haya terminado. El estado no sustituye
la comprobación de actividad del proceso.

Cada corrida guarda `status.json`, `events.jsonl`, `worker.log` cuando se usa
segundo plano, y checkpoints con pesos y estado de reanudación. Conserva el
último checkpoint coherente y el elegido por validación. Un error o cierre brusco
puede perder pasos posteriores al último checkpoint completo.

## Detener y reanudar

```bash
unidad conversation stop --run-dir private/conversation/runs/basic-v2
unidad conversation status --run-dir private/conversation/runs/basic-v2
```

`stop` solicita una parada cooperativa al proceso de esa corrida. Comprueba su
estado hasta que deje de ejecutarse y tenga un checkpoint antes de reanudar:

```bash
unidad conversation resume \
  --corpus private/conversation/basic-es-v1.jsonl \
  --run-dir private/conversation/runs/basic-v2
```

La reanudación mantiene modelo, tokenizador, optimizador, scheduler, posición en
los ejemplos, generadores aleatorios y tiempo acumulado. Exige el mismo hash del
corpus y un checkpoint íntegro; no sirve para añadir datos a mitad de una corrida.
Si ya se agotó el presupuesto total o la paciencia de validación, reanudar no
amplía esos límites. Un corpus nuevo o una etapa distinta requieren otra corrida.
Esta versión todavía no implementa inicializar esa etapa nueva desde un artefacto
anterior: `start` comienza desde pesos aleatorios y `resume` continúa exactamente
la misma corrida. La transferencia de pesos propios entre currículos queda
pendiente; no hay un ciclo automático que aprenda tareas nuevas.

El trabajador puede continuar al terminar el comando que lo lanzó. Sólo lo hace
mientras el entorno de ejecución siga activo: `environmentLifetimeGuaranteed`
es `false`. La persistencia de los archivos y la continuidad del proceso dependen
del entorno. Para continuar en otro equipo conserva el corpus, el directorio de
la corrida y una instalación compatible; no copies sólo los pesos si necesitas
reanudar el optimizador.

## Guardar una instantánea y cerrar la evaluación

El checkpoint se actualiza cuando la pérdida de validación mejora por encima del
umbral `min_delta`, nunca por test. También se evalúa la inicialización: si
entrenar no mejora la validación según ese criterio, el checkpoint elegido puede
seguir siendo el del paso cero. Para usar el seleccionado disponible sin
consultar test ni detener la corrida:

```bash
unidad conversation snapshot \
  --run-dir private/conversation/runs/basic-v2 \
  --output local-models/basic-v2-preview-01
```

Este comando llama a `export_best` y deja `testEvaluated: false` en los metadatos.
Puede ejecutarse durante el entrenamiento. Usa una ruta nueva para conservar
la identidad de cada instantánea; no cambies los archivos de un modelo cargado
por un producto.

Cuando decidas cerrar la etapa, detén el trabajador y ejecuta:

```bash
unidad conversation evaluate \
  --run-dir private/conversation/runs/basic-v2 \
  --corpus private/conversation/basic-es-v1.jsonl \
  --output local-models/basic-v2-final
```

`evaluate` obtiene el checkpoint elegido por validación, calcula NLL y perplexity
por token de respuesta en train, validación y test, y compara con un baseline
unigrama ajustado sólo con train. Escribe `evaluation.json`: desde ese momento
la corrida no se puede reanudar. Test queda fuera del bucle de entrenamiento y
de selección. No uses su resultado para retocar esta corrida; las decisiones de
desarrollo deben apoyarse en train/validación y reservar una prueba independiente
para una etapa futura.

El entrenador supervisa la última respuesta del asistente de cada registro,
incluido su final de secuencia; los turnos anteriores sirven de contexto. Divide
los diálogos largos en ventanas consecutivas sin repetir objetivos. Cada ventana
reinicia el contexto, de modo que las respuestas largas pueden perder contexto
anterior en sus límites. Los conteos de ventanas y tokens evaluados no equivalen
al número original de registros. Menor NLL no acredita que la respuesta sea útil,
correcta o capaz de seguir una instrucción.

## Probar una respuesta real

```bash
unidad conversation chat \
  --model local-models/basic-v2-preview-01 \
  --message 'Hola, hoy quiero hablar un rato.' \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --seed 42
```

La salida es JSON con el texto generado por esos pesos, `modelId`, IDs y cantidad
de tokens generados, `stopReason` y los tokens de contexto descartados. Puede
terminar por fin de secuencia o por presupuesto; una respuesta vacía, incoherente
o repetitiva sigue siendo un resultado posible. `--temperature 0` elige el token
de mayor puntuación; los valores mayores muestrean con una semilla local.

Para varios turnos, usa `--messages archivo.json` o `--messages -` con un array de
objetos `{"role": "user", "content": "texto"}` y turnos previos de `assistant`.
La última entrada debe ser `user`. Se admite un `system` inicial explícito;
si falta, se añade el mismo contexto de estilo usado al preparar el entrenamiento:

> Eres Unidad. Habla en español de forma cercana y de confianza. Relaciona lo que
> la persona ya ha compartido; distingue tus inferencias de los hechos y no
> inventes recuerdos.

Ese texto condiciona la generación; no demuestra recuerdos, comprensión ni
autoriza herramientas. Los mensajes permiten hasta 64 turnos contando el sistema,
16.384 bytes UTF-8 por mensaje y 65.536 bytes en total. Si no envías un sistema,
debes dejar espacio para el predeterminado. El contexto efectivo de la
configuración básica conserva sólo los últimos 256 tokens y la generación usa
una ventana deslizante. El máximo público de salida es 1.024 tokens.

## Usarlo desde el cerebro de un producto

El anfitrión fija el directorio del modelo al crear el cerebro. Dos productos
conectados al mismo `Brain` comparten esos pesos y su carga, y envían su contexto
por separado:

```python
from pathlib import Path
from ia_propia.brain import Brain

brain = Brain(Path("private/brain"),
              conversation_model=Path("local-models/basic-v2-preview-01"))
session = brain.connect("mi-producto")
response = session.request({
    "schemaVersion": 1,
    "operation": "conversation.reply",
    "input": {
        "messages": [{"role": "user", "content": "¿Por dónde empezamos?"}],
        "max_new_tokens": 64,
        "temperature": 0.7,
        "seed": 42
    }
})
```

La CLI admite la misma configuración mediante `--conversation-model`:

```bash
unidad brain capabilities \
  --body mi-producto \
  --state-dir private/brain \
  --conversation-model local-models/basic-v2-preview-01
unidad brain call \
  --body mi-producto \
  --state-dir private/brain \
  --conversation-model local-models/basic-v2-preview-01 \
  --input - <<'JSON'
{"schemaVersion":1,"operation":"conversation.reply","input":{"messages":[{"role":"user","content":"Hola."}],"max_new_tokens":64,"temperature":0.7,"seed":42}}
JSON
```

`brain serve` también acepta `--conversation-model` para servir esa operación
por HTTP autenticado; los detalles de cuerpos, herramientas y memoria están en
[docs/brain.md](brain.md). Las peticiones no seleccionan rutas ni sustituyen
pesos. El historial se envía explícitamente: registrar memoria no lo añade
automáticamente al contexto, ni cambia los parámetros. Sin configurar un modelo
conversacional, `conversation.reply` devuelve `capability_unavailable`.

## Portabilidad y siguientes etapas

Una exportación contiene `config.json`, `tokenizer.json`, `weights.safetensors`
y `manifest.json`. El manifiesto identifica modelo y hashes SHA-256 de los tres
archivos; la carga comprueba configuración, vocabulario, formas y valores de los
pesos. La inferencia carga safetensors y no deserializa pickle. El estado local
de reanudación del optimizador es un checkpoint de entrenamiento distinto del
artefacto que necesita un producto.

Esta etapa enseña conversación breve con pocos datos y una red pequeña. Puede
memorizar ejemplos, confundir hechos, perder contexto y producir texto defectuoso.
La CPU disponible limita el volumen de entrenamiento; más tiempo sobre los
mismos ejemplos puede sobreajustar. Los campos de validación de instrucciones,
conversación general y capacidad de trabajador siguen en `false`.

La evolución requiere etapas nuevas: ampliar y revisar conversaciones, medir
continuidad y calidad con casos independientes, y después preparar tareas de
programación con evaluación específica. Proponer cambios a su propio código
y ejecutarlos con resultados comprobados son capacidades futuras. Ni esta
pérdida de lenguaje ni conectar herramientas al cerebro las acredita.
