# Unidad: español ampliado, segunda etapa

Instantánea experimental del paso **500** de la segunda etapa. Conserva la
arquitectura de 1.088.256 parámetros y el tokenizador BPE de la primera etapa.
La inicialización reutilizó exactamente los mejores pesos propios de esa etapa
(paso 2.100); el optimizador se reinició para aprender con el corpus ampliado.

El corpus nuevo contiene 14.207 registros: 120 conversaciones sintéticas y
14.087 públicas revisadas. Conserva todos los ejemplos básicos y sus particiones;
las copias nuevas no trasladan textos reservados a entrenamiento. Sobre esta
misma validación ampliada, la pérdida NLL bajó de **5,1455** al iniciar la etapa a
**4,6214** en esta instantánea. Test no se ha evaluado.

El proceso sigue entrenándose en el entorno donde se inició, con el tiempo
restante del presupuesto original de 24 horas. La exportación es fija y no cambia
cuando ese proceso guarda otros checkpoints.

**Todavía no habla de forma competente.** Sus frases pueden mezclar temas,
inventar palabras y perder la relación con la pregunta. Menor pérdida por token
no demuestra comprensión, conversación útil ni programación autónoma.

```bash
unidad conversation chat --model models/conversation-spanish-v2 \
  --message 'Hola, ¿cómo empezamos?' --max-new-tokens 64 --seed 42
unidad conversation status --run-dir private/conversation/runs/expanded-v1
```

El manifiesto registra el modelo padre, los hashes de sus pesos y corpus, el
origen aleatorio del entrenamiento propio y los archivos de esta instantánea.
Para generar texto se necesitan los cuatro archivos de inferencia del directorio;
para reanudar el optimizador se necesita además la corrida privada y su corpus.

La fuente pública es OpenAssistant/oasst1, revisión
`fdf72ae0827c1cda404aff25b6603abec9e3399b`, con
[licencia Apache-2.0](../../docs/licenses/OASST1-Apache-2.0.txt).
No se usó este chat ni las órdenes del creador como corpus personal.
El [registro de progresión](../../docs/unidad-conversation-progression-v0.4.0.json)
conserva las verificaciones y las respuestas reales. La
[guía de conversación](../../docs/conversation.md) explica la continuación entre etapas.
