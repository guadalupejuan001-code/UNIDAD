# GNOSIS: un proyecto para el cerebro y su aprendizaje

GNOSIS reúne la selección del cerebro, NERVA y las etapas de entrenamiento en
[una configuración](../configs/gnosis.json). Conserva el mismo `Brain`, el mismo
`ConversationEngine` y el protocolo JSON que ya utilizan los productos. No añade
otro generador de respuestas ni un planificador aprendido.

La configuración fija una referencia por identidad, un perfil de arquitectura,
presupuestos de memoria y almacenamiento, y tres etapas con directorios propios:

1. `foundation`: aprender texto en español desde pesos aleatorios con `causal_lm`.
2. `conversation`: ajustar el checkpoint propio anterior con respuestas de diálogo.
3. `adaptation`: una etapa posterior con datos y correcciones explícitamente preparados.

Las etapas conservan el tokenizador y el linaje. Reanudar conserva además el
optimizador; iniciar otra etapa crea uno nuevo. Registrar memoria y conversar
no ejecuta entrenamiento. Los mensajes personales y la memoria de los productos
no se convierten automáticamente en corpus.

## Usar el proyecto

Instala las dependencias conversacionales y Node.js 18+:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[conversation]'
unidad gnosis plan --project configs/gnosis.json
unidad gnosis capabilities --project configs/gnosis.json --body pc
unidad gnosis chat --project configs/gnosis.json --temperature 0
```

`plan` no asigna pesos, no inicia procesos y no entrena. La configuración entregada
selecciona C7 para conversar y el perfil `50m` para una nueva base; los corpus
están sin configurar. El informe indica ese bloqueo. El chat sigue siendo
experimental: C7 no ha superado la aceptación conversacional.

`brain.conversation_decoder` conecta el selector `"torch"` o `"nerva"` al mismo
`ConversationEngine`; `brain.conversation_cache` habilita la caché por respuesta.
Las configuraciones anteriores que omiten estos campos conservan Torch y usan
caché. La configuración entregada elige Torch con caché por su
[mejor tiempo medido](nerva-conversation-20261013/README.md); NERVA está disponible
como selector real con Node.js 18+. `chat` admite `--top-k`, `--top-p` y
`--repetition-penalty`, con valores neutrales por defecto. Cambiar esos controles
no entrena los pesos ni demuestra mejor conversación.

Desde Python, todos los productos pueden utilizar el mismo objeto:

```python
from ia_propia.gnosis import load_project

project = load_project("configs/gnosis.json")
pc = project.connect("pc")
android = project.connect("android")
# pc.brain is android.brain; la memoria permanece separada por producto.
```

`call --input archivo.json` y `serve` reutilizan el contrato del cerebro.
`serve` necesita el token de `UNIDAD_API_TOKEN` y escucha por defecto en localhost.
La API permite un futuro cliente Android; este cambio no conecta por sí solo
la aplicación Android existente. Las peticiones no pueden cambiar modelos,
backends, rutas, presupuestos ni herramientas.

Para aprender, configura en cada etapa `corpus` y su `corpus_sha256`, conserva
la procedencia y las particiones por documento o familia, y utiliza datos que
tengas derecho a emplear. El formato de texto está descrito en
[preentrenamiento](language-pretraining.md); SFT usa el formato conversacional
existente. TEST permanece fuera del cargador de entrenamiento. Con datos y
recursos admitidos, los comandos son:

```sh
unidad gnosis train --project configs/gnosis.json --stage foundation --budget-seconds 600
unidad gnosis status --project configs/gnosis.json --stage foundation
unidad gnosis resume --project configs/gnosis.json --stage foundation --budget-seconds 600
unidad gnosis export --project configs/gnosis.json --stage foundation
unidad gnosis train --project configs/gnosis.json --stage conversation --budget-seconds 600
unidad gnosis export --project configs/gnosis.json --stage conversation
unidad gnosis assess --project configs/gnosis.json --stage conversation \
  --cases casos-nuevos.json --output evaluacion.json
```

La configuración entregada selecciona `data_loading="mmap"` en las nuevas
etapas. El cargador valida el corpus por iteraciones y prepara TRAIN/VALIDATION
en disco sin materializar todos los textos ni ventanas. No cambia los datos sin
configurar ni inicia una corrida. Consulta los
[controles y límites del modo mmap](language-pretraining.md#corpus-y-ventanas-desde-disco).
Las configuraciones y checkpoints anteriores mantienen su modo guardado.

`export` conserva el checkpoint elegido por validación en `run_dir/artifact`.
No lo convierte en referencia ni acredita conversación. `assess` conserva las
respuestas reales mediante el motor común; para aceptar un candidato hacen falta
la rúbrica y revisión de conversaciones nuevas, además de las verificaciones
técnicas. La referencia de inferencia debe permanecer en un directorio separado
de las corridas y se selecciona explícitamente mediante su identidad.

## Arquitecturas y límites reales

| Perfil | Parámetros nominales | Contexto configurado | Comprobación actual |
| --- | ---: | ---: | --- |
| `small` | 6.967.296 | 512 | Arquitectura pequeña ya disponible |
| `50m` | 48.852.992 | 1.024 | Modelo real, forward/backward de 64 tokens, exportación y carga |
| `100m` | 99.446.400 | 1.024 | Conteo y validación; no entrenado ni ejecutado aquí |
| `1b` | 991.586.304 | 2.048 | Conteo y estructura meta; no entrenado ni ejecutado aquí |

Los conteos nominales usan vocabularios de 8.192 y 16.384 tokens. Un tokenizador
ajustado con poco texto puede tener menos entradas. La admisión de una etapa
heredada o reanudada usa su configuración real antes de cargar los tensores,
rechaza arquitecturas ajenas al perfil y fija esa configuración en la carga.

Las configuraciones antiguas conservan el límite de 20 millones de parámetros.
Un `parameter_budget` explícito permite hasta 1.200 millones. Este campo no aporta
pesos entrenados, GPU ni calidad. Las estimaciones de recursos incluyen parámetros,
gradientes, AdamW, activaciones, proyección, datos en RAM y margen; no son una
garantía del pico real. Se verifican también espacio libre e identidad del corpus.
Las rutas de las etapas no pueden anidarse ni invadir memoria o referencia.
La admisión conserva la reserva conservadora de memoria para datos incluso con
mmap, porque el tokenizador, metadatos y orden aún consumen RAM; no interpreta
el ahorro de una medición como garantía para cualquier corpus. Añade una
estimación de almacenamiento de la caché. Estas heurísticas por tamaño de archivo
no garantizan el espacio de corpus comprimidos ni sustituyen la medición real.

## Optimizaciones comprobadas

`micro_batch_size` divide un lote lógico sin almacenar todas sus activaciones a
la vez. La pérdida pondera correctamente todos los tokens o todas las secuencias;
el clipping, optimizador y scheduler avanzan una sola vez por lote lógico.
La reanudación de la misma política se comprobó exactamente en FP32 y BF16,
incluido un fallo en mitad de un lote. No se promete identidad bit a bit frente
al lote completo: orden de sumas y consumo de RNG pueden variar. Por ahora no se
combina con auxiliares de copia ni muestreo adaptativo.

En una fixture pequeña de ocho secuencias, las reservas únicas de activaciones
retenidas por autograd bajaron de 21.518.340 a 2.698.760 bytes, un 87,46 %, con
la misma pérdida y una diferencia máxima de gradientes de `7,92e-9`. Es una
medida de esas reservas, no de RAM total, velocidad ni conversación.

Los artefactos ahora se escriben directamente y se verifican por bloques.
La carga usa una estructura meta y una copia CPU fija, evitando crear otro
modelo aleatorio y conservar el archivo entero en bytes. Cambiar el archivo
posteriormente no cambia los pesos del motor ya cargado.

La [medición de escalado](gnosis-scaling-20261013.json) utilizó seis procesos
independientes, CPU con un hilo y un modelo aleatorio de 48.852.992 parámetros:

| Operación | Pico RSS nuevo | Pico RSS del serializador anterior |
| --- | ---: | ---: |
| Exportación | 651 MB | 1.237 MB |
| Carga | 710 MB | 876 MB |

Son observaciones únicas bajo condiciones declaradas. El RSS residente final no
mejoró en la carga; no se atribuye ahorro universal ni velocidad. Ambos caminos
produjeron archivos, tensores y logits idénticos. C7 mantuvo sus cuatro archivos
e identidad originales. El [runner](../examples/verify_gnosis_scaling.py) permite
reproducir el contraste usando una copia del serializador anterior.

```sh
PYTHONPATH=src python examples/verify_gnosis_scaling.py \
  --output private/gnosis-benchmark \
  --legacy-artifacts docs/gnosis-evidence/artifacts-before-stream.py \
  --report private/gnosis-benchmark/report.json
```

## Prioridad para acercarse a una IA útil

El modelo de 49 millones de la prueba tiene pesos aleatorios y cero actualizaciones;
no ha aprendido conversación. El contexto completo de 1.024 tokens y el optimizador
de ese modelo tampoco se probaron en ese benchmark. NERVA ejecuta el char-RNN y
ahora puede seleccionar los tokens del Transformer. PyTorch conserva el cálculo
neuronal del Transformer; no se ha traducido ese cálculo a NERVA.

La siguiente prioridad es ampliar y diversificar la base lingüística de español,
con entrenamiento verificable y diálogos nuevos que midan conservación de hechos
y correcciones. Aumentar parámetros antes de resolver datos y generalización no
acredita acercamiento conversacional a asistentes comerciales. El cargador mmap
opcional resuelve la retención de textos y ventanas en RAM para muestreo uniforme;
el [contraste de preparación](tokenized-data-20261013.md) documenta su efecto.
Continúan pendientes entrenamiento con GPU y aprendizaje conversacional aceptado.
No hay autoaprendizaje,
autoprogramación ni conversación competitiva demostrados por esta entrega.

