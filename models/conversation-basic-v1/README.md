# Unidad: primera instantánea conversacional

Modelo experimental propio de 1.088.256 parámetros. Transformer causal de cuatro
capas, dimensión 128, cuatro cabezas, vocabulario BPE de 2.048 tokens y contexto
de 256 tokens. Se entrenó desde inicialización aleatoria; no usa pesos descargados
ni llamadas a otra IA.

Esta instantánea fija conserva el checkpoint del paso **1.600**, seleccionado por
validación. La pérdida NLL por token de respuesta bajó de **7,6471** en la
inicialización a **5,0347**. El conjunto de prueba no se ha evaluado: el trabajo
local continúa y esta exportación no lo detiene ni lo sella.

**Sus respuestas todavía son defectuosas.** Una salida real a «Hola, ¿cómo
empezamos?» fue: «Claro, por lo que te puedo ayudar a lo que quieres que te al
través de..., la sol.» La menor pérdida no demuestra comprensión, seguimiento de
instrucciones ni programación autónoma.

El corpus contiene 120 conversaciones sintéticas identificadas y 1.087 registros
públicos revisados de OpenAssistant/oasst1, revisión
`fdf72ae0827c1cda404aff25b6603abec9e3399b`, Apache-2.0. Las familias están separadas
entre entrenamiento, validación y prueba. No se usaron las conversaciones privadas
del creador ni sus órdenes como corpus. La licencia de la fuente pública está
en [OASST1-Apache-2.0.txt](../../docs/licenses/OASST1-Apache-2.0.txt).

Los cuatro archivos de inferencia son `config.json`, `tokenizer.json`,
`weights.safetensors` y `manifest.json`. El manifiesto identifica y verifica sus
bytes mediante SHA-256. No se necesita el corpus para generar texto. Esta
exportación no incluye el estado del optimizador necesario para reanudar el
entrenamiento; los checkpoints de la corrida permanecen en el directorio privado
del entorno que la ejecuta.

```bash
unidad conversation chat --model models/conversation-basic-v1 \
  --message 'Hola, ¿cómo empezamos?' --max-new-tokens 64 \
  --temperature 0.7 --seed 42
```

El [registro de evidencia](../../docs/unidad-conversation-validation-v0.3.0.json)
conserva los hashes, configuración, entrenamiento, respuestas y pruebas. La
[guía de conversación](../../docs/conversation.md) explica cómo entrenar y conectar
estos pesos al cerebro de un producto.
