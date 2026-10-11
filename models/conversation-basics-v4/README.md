# Unidad: cuarta instantánea de desarrollo conversacional

Este artefacto conserva **progreso intermedio experimental**. El objetivo de
conversación básica sigue pendiente. Se copiaron sin cambios los cuatro archivos
de inferencia del checkpoint seleccionado en el paso **250** de la etapa natural;
no se generaron nuevas respuestas para preparar esta exportación.

La red tiene **1.088.256 parámetros**, cuatro bloques de dimensión 128 y contexto
de 256 tokens. Continúa los pesos propios de la etapa abierta, seleccionados en
su paso 2.000; su origen inicial sigue siendo aleatorio. Se conserva el BPE propio
de 2.048 tokens. No se usaron pesos externos ni llamadas a otra IA para responder.
El optimizador y el calendario se reiniciaron al comenzar esta etapa.

El corpus combinado tiene **24.498 registros**: 19.178 de entrenamiento, 2.649 de
validación y 2.671 de test. Incluye 23.248 ejemplos ficticios identificados y 1.250
diálogos públicos revisados de OpenAssistant/OASST1. Se usó pérdida media por
secuencia y muestreo ponderado; se repitieron veinte aperturas breves distintas,
sin convertir esas repeticiones en ejemplos nuevos. La NLL de validación del
checkpoint seleccionado es **0,4029**; esa cifra no demuestra conversación útil.

En veinte plantillas de desarrollo de validación, con tres turnos y respuestas
anteriores **realmente generadas**, hubo **18/60 coincidencias exactas**. Otro
asistente del proyecto revisó semánticamente los sesenta resultados: **27
completos, 7 parciales y 26 fallidos**. Sólo **4/20 sesiones** cumplieron sus tres
turnos. Esa revisión no fue ciega ni realizada por un panel humano externo.

El primer saludo funciona en el diálogo libre guardado. La identidad del
asistente después del saludo y los nombres nuevos aún fallan. Esta secuencia
ficticia conserva las respuestas originales, sin corregirlas:

| Mensaje del usuario, en orden | Respuesta real |
| --- | --- |
| Hola, Unidad. | Hola. ¿Qué tienes en mente? |
| ¿Cómo te llamas? | Hola, Kerauna. ¿Qué tienes en mente? |
| Me llamo Yara. | Hola, Kerauna. ¿Cómo estás? |
| ¿Cómo me llamo? | Te llamas Kerauna. |

También permanecen errores de hechos y correcciones, respuestas incompletas y
repeticiones. **Conversación básica no aceptada; conversación general no
validada.** Test y la aceptación prospectiva final siguen sin evaluarse. No se
acreditan conciencia, autonomía para trabajar ni modificación propia de código.

La [evidencia de desarrollo](../../docs/unidad-conversation-basics-v4-development.json)
incluye las sesenta respuestas con historial y tokens, todos los juicios de la
rúbrica y los cinco diálogos libres ficticios. Ningún chat del creador se exportó
como corpus o evidencia. Las coincidencias exactas y la revisión semántica se
registran por separado, con su procedencia y límites.

```bash
unidad conversation chat --model models/conversation-basics-v4 \
  --message 'Hola, Unidad.' --max-new-tokens 32 --temperature 0 --seed 42
```

La carga mediante `ConversationEngine` desde esta ruta verificó la identidad
`unidad-conversation-cc55bb398d3aa188d4105e2dca63acae2c888a2a3af1386fc40465e7c7d44fa0`.
Los SHA-256 de los pesos, BPE y corpus son, respectivamente:

- `e77d57c13af16698743efaed680c8772479445610071083e1868c619237156ae`
- `ee3b0f263ce97919f0a6b6ea8bd1329d1a008883116ee3a2decc9eb15f6fbb59`
- `34931328839e47949d21ddb977ea1f58fcff432e9e428e80413ac31093811647`

`config.json`, `tokenizer.json`, `weights.safetensors` y `manifest.json` permiten
inferencia local. Reanudar exactamente el entrenamiento requiere además el
corpus y los estados privados del optimizador y del checkpoint; no están dentro
de estos cuatro archivos. El manifiesto conserva los indicadores experimentales
y `productionAccepted=false`.

La fuente pública OASST1 conserva su
[licencia Apache-2.0](../../docs/licenses/OASST1-Apache-2.0.txt).
