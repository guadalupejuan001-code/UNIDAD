# NERVA dentro del cerebro de Unidad

Unidad puede ejecutar su **char-RNN propio** mediante NERVA 0.2. El programa
`src/ia_propia/assets/nerva/language_rnn.nv` calcula la recurrencia `tanh`, la
proyección de salida y cada decisión greedy en tensores `f64`. Python valida el
artefacto original y conserva su tokenizador, identidad, procedencia y etiquetas
experimentales. No se cambian los pesos ni se llama a otro modelo.

Esta conexión no ejecuta el Transformer conversacional C7 ni mejora por sí misma
sus respuestas. NumPy sigue siendo el motor predeterminado. NERVA se selecciona
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

Por ahora sólo se admite generación greedy con temperatura cero. El CLI de
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
