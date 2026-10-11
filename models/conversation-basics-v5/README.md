# Bases y recuperación experimental de Unidad v5

Esta instantánea contiene pesos propios de Unidad, derivados de su Transformer
entrenado desde pesos aleatorios. No llama a otro modelo. Usa el mismo motor
conversacional desde consola, Python y HTTP.

La etapa añadió diversidad de formulaciones y errores ficticios en respuestas
anteriores. Sólo la última respuesta correcta y EOS fueron objetivos de
entrenamiento. Se mantuvieron los datos y particiones anteriores, el tokenizador
BPE de 2.048 tokens y los 1.088.256 parámetros.

Se conserva la instantánea del paso 500 de `natural-recovery-v2`, exportada cuando
era el mejor checkpoint de validación disponible. Las copias posteriores bajaron
la pérdida pero no conservaron mejor los hechos en la conversación de desarrollo.
Esta elección usa desarrollo; no es una selección independiente por test.

En los mismos 22 turnos de desarrollo, con las respuestas anteriores realmente
generadas, obtuvo 13 respuestas completas, 2 parciales y 7 fallidas; 2 de 5
episodios completos. El padre obtuvo 6 completas. Recuperó el apodo solicitado
y el cambio de color de una taza, aunque las respuestas de color tienen un error
menor de concordancia. Falló con ciudades, repitió sílabas en un cambio de tema
y respondió mal a algunas peticiones. La revisión fue de dos asistentes del
proyecto, sin cegamiento; no se atribuye revisión humana o externa.

```bash
unidad conversation converse --model models/conversation-basics-v5 \
  --temperature 0 --max-new-tokens 32
```

Modelo: `unidad-conversation-04c65431b87bff6e9877611cefeff883542b0a0157dc962d144826944f6a54b9`.
SHA-256 de los pesos:
`c564039ba0d6d3bf681cc8a7c76555de91b8fbd6c8fc59fa5120889f13323e05`.

Los cuatro archivos del artefacto se copiaron sin modificar sus bytes ni su
manifiesto. El [registro de aprendizaje](../../docs/unidad-learning-curriculum-v2.json)
conserva datos, pruebas y referencias verificables de los resultados.

Continúa siendo experimental: no se ha aceptado como conversación competente,
no se evaluó el test final y no aprende ni modifica su código durante la charla.
