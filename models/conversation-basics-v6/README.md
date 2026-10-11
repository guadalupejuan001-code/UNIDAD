# Refinamiento conversacional experimental de Unidad v6

Esta instantánea conserva pesos propios de Unidad, derivados de su Transformer
entrenado desde pesos aleatorios. Mantiene 52 tensores nativos, 1.088.256
parámetros, el tokenizador BPE de 2.048 tokens y el contexto de 256 tokens. No
llama a otro modelo. Se usa mediante el mismo `ConversationEngine` de consola,
Python y HTTP, seleccionando este directorio explícitamente.

La nueva etapa partió del [modelo v5](../conversation-basics-v5/README.md), con
reinicio del optimizador AdamW y del plan de tasa de aprendizaje. Redujo la tasa
a `0.0001` y cambió conjuntamente los factores de muestreo: OpenAssistant `12`,
flujo variable `3`, diversidad nueva `2`, copia literal `0.5`, recuperación `3`
y bases naturales `2`, con un factor adicional `16` para sus primeros turnos.
Conservó el corpus y sus particiones; sólo la última respuesta y EOS fueron
objetivos supervisados. No usó práctica adaptativa ni evaluó el test final.
Como cambiaron la tasa y la mezcla a la vez, el resultado no identifica cuál de
ellas causó la diferencia.

La corrida terminó 1.200 pasos en 333,2478 segundos, incluyendo preparación
y validación. Se conserva el checkpoint nativo del paso 600, seleccionado por
validación, cuya pérdida media por ventana fue
`0.37595150808145594`, frente a `0.47535643555308` al inicio. Esa pérdida no
acredita por sí sola conversación competente.

En los mismos 22 turnos de desarrollo, usando las respuestas anteriores
realmente generadas, obtuvo 15 respuestas completas, ninguna parcial y 7
fallidas; completó 2 de 5 episodios. V5 obtuvo 13 completas, 2 parciales y 7
fallidas, también 2 episodios completos, y se conserva sin cambios. V6 falló
los cuatro turnos de ciudades y los dos de Mapa; también respondió mal sobre
una bicicleta cuyo color no se había indicado. Es una sonda conocida y repetida
de desarrollo, revisada por dos asistentes del proyecto sin cegamiento. No hubo
revisión humana o externa ni evaluación independiente por test final.

```bash
unidad conversation converse --model models/conversation-basics-v6 \
  --temperature 0 --max-new-tokens 32
```

Modelo: `unidad-conversation-4708a11a968afeacf4f130632865dd376452a7fd7c060736b2f8da7bdc795d04`.
SHA-256 de los pesos:
`6bb2128334dc701cf9f97d2c6501d4761b89ef5b0c149bdd3439f89976d01dc2`.

Los cuatro archivos del artefacto se copiaron sin modificar sus bytes ni su
manifiesto. El [registro de refinamiento](../../docs/unidad-learning-refinement-v0.5.2.json)
conserva los resultados y su procedencia verificable.

Esta instantánea es una opción experimental: no ha sido aceptada como
conversación competente o como trabajador autónomo. No aprende ni modifica su
código durante la charla.
