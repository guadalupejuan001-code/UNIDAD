# Unidad: diálogo con contexto, tercera etapa

Instantánea experimental del paso **200**, seleccionada por validación. Reutiliza
los pesos propios del paso 7.500 de la etapa de español ampliado y el mismo BPE,
con origen inicial aleatorio registrado. No utiliza otro modelo ni una API de IA.
Se conservaron exactamente los 52 tensores al iniciar la etapa; el optimizador
y el calendario se reiniciaron.

El corpus de esta etapa tiene 1.075 diálogos completos: 660 ficticios identificados
y 415 públicos revisados de OASST1. Se separa por familias: 885 entrenamiento,
88 validación y 102 test. Todos caben junto con el sistema predeterminado en el
contexto de 256 tokens. La política `sliding_context` conserva contexto causal
y no repite objetivos de aprendizaje. La red mantiene 1.088.256 parámetros.

Sobre la misma validación, la NLL bajó de **3,9018** a **3,7806**. Después empeoró
mientras seguía bajando el error de entrenamiento; se pausó la corrida y se
conservó esta instantánea anterior. Test sigue sin evaluarse. La etapa posterior
añade variedad de diálogos breves y reduce la tasa de aprendizaje.

**Todavía no conversa de forma competente.** En los 14 pares de desarrollo,
esta instantánea no prefirió ambas referencias en ningún par; sus respuestas
siguen siendo genéricas, incorrectas o incomprensibles. El motor compartido y
las pruebas técnicas acreditan integración, no comprensión ni autoprogramación.

```bash
unidad conversation chat --model models/conversation-context-v3 \
  --message 'Hola.' --max-new-tokens 64 --seed 42
unidad conversation assess --model models/conversation-context-v3 \
  --cases examples/conversation-evaluation.json \
  --output private/conversation/context-v3-development.json
unidad conversation status --run-dir private/conversation/runs/balanced-short-v1
```

Los cuatro archivos de inferencia contienen configuración, tokenizador, pesos
safetensors y manifiesto con identidad y SHA-256. La exportación es fija.
La reanudación requiere el corpus y checkpoints privados del entorno original.

La fuente pública es OpenAssistant/oasst1, revisión
`fdf72ae0827c1cda404aff25b6603abec9e3399b`, con
[licencia Apache-2.0](../../docs/licenses/OASST1-Apache-2.0.txt). Los ejemplos
nuevos son ficticios; no se usó el chat del creador como corpus.
El [registro de integración](../../docs/unidad-conversation-cohesion-v0.5.0.json)
conserva el diagnóstico, las etapas y los resultados reales.
