# NERVA dentro del cerebro de Unidad

## Selección de tokens del Transformer

NERVA está conectado al `ConversationEngine`: el programa fijo
`conversation_decoder.nv` recibe los logits del modelo propio y elige el
siguiente token. El cálculo neuronal permanece en PyTorch. Consola, Brain/API,
GNOSIS y evaluación reutilizan este mismo recorrido.

```sh
unidad conversation chat --model models/conversation-reference-c7 \
  --decoder-backend nerva --temperature 0 --max-new-tokens 160 \
  --message "Hola"
```

En Python: `ConversationEngine(ruta, decoder_backend="nerva")`; en Brain:
`Brain(estado, conversation_model=ruta, conversation_decoder="nerva")`.
El CLI de Brain admite `--conversation-decoder nerva`. En GNOSIS se configura
`brain.conversation_decoder` como `"nerva"`; `"torch"` es la opción predeterminada
más rápida en la medición actual. `brain.conversation_cache` controla la caché.

El selector admite temperatura 0–2, `top_k` 0–65536, `top_p` 0,000001–1 y
`repetition_penalty` 1–2. Temperatura cero elige el token permitido de mayor
puntuación. Los controles neutrales son `top_k=0`, `top_p=1`,
`repetition_penalty=1`; la penalización afecta sólo a los tokens ya generados en
esa respuesta. EOS sigue permitido. Los mismos controles están disponibles en
Torch y en el contrato de `conversation.reply`.

NERVA reutiliza un proceso Node por respuesta, con entrada acotada, programa
fijo y tiempo máximo por decisión. Cierra el proceso y sus archivos temporales
al terminar o fallar; nunca cambia silenciosamente a Torch. Los mensajes no
pueden aportar programas, rutas ni comandos. Usa un RNG local con la semilla de
la petición; en muestreo, una misma semilla se reproduce dentro de cada backend,
pero no promete los mismos tokens entre Torch y NERVA porque sus RNG difieren.

La [medición y sus respuestas reales](nerva-conversation-20261013/README.md)
conservan igualdad greedy en diez consultas C7. NERVA añadió tiempo; la mejora
de velocidad procede de la caché Torch. Seleccionar tokens es un mecanismo
estándar de generación, sin aportar por sí mismo comprensión ni conocimientos.

## Modelo de caracteres: conexión anterior conservada

Unidad puede ejecutar su **char-RNN propio** mediante NERVA 0.2. El programa
`src/ia_propia/assets/nerva/language_rnn.nv` calcula la recurrencia `tanh`, la
proyección de salida y cada decisión greedy en tensores `f64`. Python valida el
artefacto original y conserva su tokenizador, identidad, procedencia y etiquetas
experimentales. No se cambian los pesos ni se llama a otro modelo.

Esta conexión del char-RNN no ejecuta el Transformer conversacional C7 ni mejora
sus respuestas. NumPy sigue siendo el motor predeterminado del char-RNN. NERVA se selecciona
expresamente por el host y necesita **Node.js 18 o posterior**, sin paquetes npm,
red ni servicios adicionales durante la inferencia.

Desde el repositorio:

```sh
PYTHONPATH=src python -m ia_propia.language generate \
  --model src/ia_propia/assets/language-demo.json \
  --backend nerva --temperature 0 --prompt "Hola" --length 64
```

Desde Python, el mismo cerebro utiliza el motor elegido para todos sus cuerpos:

```python
from ia_propia.brain import Brain

brain = Brain("private/brain", language_backend="nerva")
body = brain.connect("mi-producto")
result = body.request({
    "schemaVersion": 1,
    "requestId": "nerva-demo",
    "operation": "language.generate",
    "input": {"prompt": "Hola", "length": 64, "temperature": 0},
})
```

Los comandos `unidad brain capabilities`, `call` y `serve` admiten
`--language-backend nerva`. Una petición no puede elegir otro motor, ejecutable,
archivo de pesos ni ruta. Las capacidades anuncian temperatura cero y el
resultado identifica `executionBackend: "nerva-0.2"`. El modo conversacional
continúa usando el mismo `ConversationEngine` y su modelo configurado por el host.

En el char-RNN sólo se admite generación greedy con temperatura cero. El CLI de
lenguaje conserva el valor predeterminado anterior de `0.8`, por lo que seleccionar
NERVA allí requiere escribir `--temperature 0`. El cerebro elegido con NERVA
usa cero cuando la petición omite la temperatura. No se usa NumPy como fallback
si falta Node, vence el timeout o se agotan los límites del intérprete.

El runtime canónico conserva sus límites: 20 millones de operaciones tensoriales,
200 mil pasos y límites de entradas y serialización. Un artefacto válido o una
generación larga puede excederlos; el adaptador informa el fallo. La generación
usa un directorio temporal privado que se elimina al terminar, sin persistir los
pesos transferidos. El host carga los pesos originales una vez y conserva su
selección; cada operación NERVA inicia un proceso Node.

La [verificación reproducible](unidad-nerva-backend-v1.json) comparó siete prompts,
incluidos vacío y caracteres desconocidos, con 64 caracteres generados cada uno:
los 448 caracteres y los campos originales del resultado coincidieron. Las
pruebas cubren además longitud cero, semillas grandes, validación del artefacto,
el desbordamiento del logit UNK excluido y errores del runtime. La coincidencia
de `f64` es empírica; acumulación y `tanh` pueden diferir de NumPy cerca de un empate.

Las siete llamadas observadas incluyeron transferencia y arranque de Node; NERVA
tardó más que NumPy. No se atribuye una mejora de velocidad, memoria, energía ni
calidad conversacional. Los pesos y el checkpoint conversacional permanecen
sin promoción por esta comprobación técnica.

Para regenerar la evidencia:

```sh
PYTHONPATH=src python examples/verify_nerva_backend.py \
  --report docs/unidad-nerva-backend-v1.json
PYTHONPATH=src python -m unittest discover -s tests -p 'test_nerva_*.py'
```

