# Unidad E3: candidato experimental de disponibilidad inicial

Pesos propios continuados desde C7. Esta versión no acredita conversación básica ni competitiva. Se conserva el candidato real de 3.072 pasos junto con las versiones anteriores.

La intervención equilibra preguntas iniciales con información presente o ausente, conserva los diálogos de cuatro turnos y usa el mismo modelo, tokenizador y `ConversationEngine`. Las respuestas reales y el historial completo están en [evidence/dialogues-actual.txt](evidence/dialogues-actual.txt) y [JSON](evidence/dialogues-actual.json). Persisten frases incoherentes, confusión de personas y lugares y fallos de corrección.

Desde la raíz del proyecto con sus dependencias de conversación:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-initial-availability-e3 --max-new-tokens 32 --temperature 0 --seed 42
```

`/salir` termina y `/reiniciar` borra el historial de esa sesión. El comando carga exactamente los pesos locales de este artefacto; no usa un asistente externo. La referencia C7 permanece disponible por separado. Las revisiones del proyecto son de agentes de IA, no evaluación humana independiente.

Modelo: unidad-conversation-fb2c43de68b8f0fe3ff1b559de4c0931345fe6d6576ab3f05419bf50618422ab. Pesos, configuración y tokenizador quedan identificados por SHA-256 en `manifest.json`. El checkpoint nativo completo con optimizador y estado se conserva en la ruta privada de E3 para continuar sin perder el trabajo.

En los 40 turnos de desarrollo conocido, E3 obtuvo **C3/P5/F32** en la [revisión primaria](evidence/review-primary.json) y **C3/P3/F34** en la [cruzada](evidence/review-cross.json): **0 de 10 diálogos completos y 12 fallos críticos** en cada revisión. No alcanza 36 respuestas completas, 8 diálogos completos ni cero fallos críticos. C7 sigue siendo la mejor referencia propia de esta comparación (7 respuestas completas), también sin aceptación. Superar ese criterio básico sería necesario, pero no suficiente para afirmar conversación competitiva.

La [verificación de consola](evidence/cli-receipt.json) terminó con código 0 y conserva cuatro respuestas reales exactamente iguales a las del candidato evaluado. El [informe de la ampliación E2/E3](../../docs/unidad-conversation-supplement-e2-e3-results.json) conserva el presupuesto, los resultados intermedios, las revisiones y la referencia externa limitada. E3 terminó 3.072 pasos, preservó el checkpoint nativo y el optimizador; no queda entrenamiento activo.
