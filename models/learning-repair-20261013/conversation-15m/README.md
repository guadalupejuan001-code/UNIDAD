# Ajuste conversacional propio de 15M: exportación intermedia

`intermediate-step004096/` conserva exactamente la primera exportación de
inferencia de `conversation-15m`, seleccionada en el paso 4.096 por validación
de desarrollo durante la pausa natural de la primera invocación. Es experimental;
la etapa continuó después de esa pausa y esta exportación no representa su
cierre ni la aceptación de conversación.

El modelo mantiene los 15.179.520 parámetros y el tokenizador BPE de 2.048
tokens de la base propia `foundation-15m`, con contexto de 512 posiciones. Se
ajustan todos los parámetros usando únicamente la última respuesta del
asistente como objetivo, pérdida `sequence_mean` y el corpus congelado
`4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510`.
No usa pesos preentrenados ajenos ni respuestas sustituidas en inferencia.

La exportación contiene cuatro archivos originales, con **60.784.882 bytes**.
Identifica el modelo
`unidad-conversation-072eedd276160f7581c5a70b57e66ae3000d2e664ed32f6b5a4669d4d12dcf57`.
La NLL seleccionada fue **1,7110848086831454**. La contabilidad de TRAIN
registra **2.638.841 objetivos supervisados** presentados hasta ese checkpoint,
mediante 32.764 ventanas; este recuento incluye EOS y no convierte el contexto
en objetivos positivos.

El [recibo de entrega](../../../docs/learning-repair-20261013/conversation-15m-intermediate-step004096-delivery.json)
conserva hashes y tamaños iguales antes de copiar, en la copia y después de
copiar. El [recibo de exposición](../../../docs/learning-repair-20261013/sft-exposure-step004096.json)
conserva la permutación y las máscaras reales usadas en el recuento. Las
[partes del checkpoint nativo](../../../checkpoints/learning-repair-20261013/conversation-15m/README.md)
permiten recuperar también Adam, scheduler y RNG; los cuatro archivos de esta
exportación sirven para inferencia y no bastan para reanudar exactamente.

```bash
python -m ia_propia.conversation chat \
  --model models/learning-repair-20261013/conversation-15m/intermediate-step004096 \
  --message "Hola, ¿cómo estás?" --temperature 0 --max-new-tokens 160 --seed 42
```

La entrega de bytes no ejecutó inferencia, cargó pesos ni evaluó TEST. La
evaluación con `ConversationEngine` y respuestas realmente generadas se registra
por separado. Una menor pérdida no acredita respuestas naturales, útiles o
coherentes, y esta exportación no sustituye C7 ni supera automáticamente el
criterio de 36/40 respuestas, 8/10 diálogos y cero fallos críticos o técnicos.
