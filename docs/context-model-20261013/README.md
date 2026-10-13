# Contexto independiente integrado en SOVENGI

Se implementó y entrenó un selector propio de fuentes, con almacenamiento
incremental y una conexión reutilizable antes de `ConversationEngine`. El
generador conserva sus pesos y su tokenizador. La integración está activada en
SOVENGI; Brain directo conserva su contrato y puede reutilizar `ContextManager`
desde el adaptador de cada producto.

## Selección y procedencia

El [modelo independiente](../../models/context-reference-v1/README.md) tiene
21 parámetros aprendidos, 20 características léxicas y temporales, y un artefacto
JSON empaquetado. Conserva texto original, rol, orden, ID y SHA-256. Selecciona
fuentes de la persona; los mensajes del asistente sólo son historial reciente
con su rol original. No transforma salidas propias en recuerdos aceptados ni
reescribe hechos como resúmenes generados.

El entrenamiento propio completó 2.000 épocas en 7,218 segundos. Las entidades
TRAIN/validación/TEST están separadas; las plantillas son ficticias y conocidas.
Los pesos se fijaron antes de TEST. La selección top1 fue **48/48 frente a
28/48** de la referencia léxica con recencia, y recall@3 **48/48 frente a 44/48**.
El [corpus](../../models/context-reference-v1/corpus.json),
[recibo](../../models/context-reference-v1/training.json),
[checkpoint](../../models/context-reference-v1/checkpoint.json) y
[selecciones](../../models/context-reference-v1/evaluation.json) permiten
revisar y reproducir esa prueba. No es una aceptación de conversación general.

## Escalado medido

SQLite usa WAL, transacciones de hasta 64 mensajes e índices por producto y
conversación. FTS recupera candidatos acotados sin ordenar todas las fuentes
comunes por BM25. Se priorizan coincidencias completas y se amplía con
coincidencias por término; el ranker recibe como máximo 48 fuentes y el sistema
selecciona como máximo ocho. No se carga todo el historial en memoria.

El [benchmark](benchmark.json) mide procesos nuevos con datos ficticios:

| Fuentes | Búsqueda + ranking + selección, p50 | Preparación, p50 |
| ---: | ---: | ---: |
| 1.000 | 1,80 ms | 3,55 ms |
| 10.000 | 2,92 ms | 4,43 ms |
| 100.000 | 2,70 ms | 5,42 ms |

En 100.000 fuentes, la ingesta fue de 30.752 fuentes/s, el p95 de búsqueda y
selección de 3,30 ms y el p95 de preparación de 7,18 ms. El proceso alcanzó
19,5 MiB de RSS. El ID relevante se seleccionó en 30/30 consultas por escala,
también mediante `ContextManager.prepare`. La recuperación relevante coincidió
en 5/5 consultas entre FTS y el fallback en un corpus pequeño.

La preparación medida utiliza un contador de tokens ficticio explícito:
incluye ingesta, recuperación, ranking y serialización, pero **excluye BPE,
PyTorch y generación**. En SOVENGI el presupuesto se comprueba con el
tokenizador real. Estos tiempos pertenecen a esta CPU y estas fuentes; no
prometen el mismo rendimiento con textos más largos o en otra máquina.

## Integración y límites

El gestor guarda sólo el sufijo nuevo de los historiales completos o rodantes,
recupera turnos recientes al continuar una sesión y reconoce reintentos de
instantáneas de varios turnos. Una petición con sólo un mensaje repetido se
trata como un nuevo turno: su intención es ambigua sin un historial o un ID
de petición estable. La conexión y sus herramientas las configura el anfitrión.

Se conserva el mensaje de sistema y la petición actual completos, se recuperan
fragmentos antiguos relacionados y se reserva espacio para generar. Si el
sistema y el mensaje actual exceden el presupuesto, SOVENGI responde con
`context_budget` antes de generar. La ventana aprendida sigue siendo de 2.048
tokens; procesos extensos necesitan mensajes acotados y recuperación, no
enviar un documento ilimitado en una sola petición.

Cada conversación tiene un ámbito privado en SQLite. Por defecto se crea uno
nuevo al conectar; `--context-session nombre` permite recuperarlo. `/nuevo`
borra sólo ese contexto y conserva las notas guardadas mediante `/recordar`.
El selector no modifica parámetros al conversar. Los historiales personales no
forman parte del corpus publicado.

La recuperación sigue siendo léxica: la búsqueda distingue tildes, puede incluir
distractores y puede omitir coincidencias parciales antiguas cuando todos sus
términos son frecuentes. Sin FTS, el fallback recorre como máximo las últimas
4.096 fuentes. Se omiten textos completos que no caben en el presupuesto de
contexto; no se cortan silenciosamente negaciones o correcciones. El almacén en
disco crece con el historial: esta entrega no acredita una política de retención
para historiales ilimitados.

## Uso y reproducción

```bash
python -m pip install -e '.[conversation]'
sovengi chat
sovengi --context-session trabajo chat
sovengi --no-context chat
python examples/benchmark_context.py --output private/context-benchmark.json
```

Desde Python, `Sovengi.connect(context_enabled=True,
context_session_id="trabajo", context_model=None)` usa el modelo empaquetado.
Un artefacto alternativo sólo se elige desde el anfitrión. El resultado incluye
`conversationContext` con IDs, hashes, presupuesto y estado, sin volver a
exponer todo el texto privado en los metadatos.

Las pruebas cubren presupuesto real, conservación de fuentes, correcciones,
aislamiento, reintentos, reapertura, roles, migración, concurrencia, checkpoints
y límites. Sus fixtures pequeños comprueban software; las respuestas con los
pesos publicados se evalúan por separado.

La [verificación local](verification.json) registra **703 pruebas aprobadas y
una omitida** en la regresión de 704, además de 12 pruebas de ejemplos. Un wheel
de `unidad 0.7.0` pasó otras 27 pruebas del selector sin importar NumPy ni
PyTorch. Incluye hashes de fuentes, artefactos y pesos del generador, además de
la comprobación de enlaces.

## Respuestas reales de SOVENGI

Se generaron 48 respuestas reales: 12 casos reservados, dos brazos por caso y
dos modelos publicados. La [referencia actual](generation-reference.json) y
el [candidato de memoria](generation-memory-candidate.json) conservan entradas
exactas, respuestas completas, identidad, hashes, parámetros y tokens después
del recorte del motor. No se entrenó ninguno de esos generadores en esta entrega.

En ambos modelos, las fuentes requeridas pasaron de **0/12 sin gestor a 9/12
con gestor**. Los tres casos de dato lejano, los tres de corrección y los tres
de objetivo/restricción conservaron sus fuentes. Los **tres antecedentes cortos
fallaron**: la recuperación léxica no resolvió la referencia al asunto antiguo.
Con el gestor, sistema y petición actual quedaron completos y el prompt cabía
con la reserva de respuesta en 12/12 casos, sin tokens descartados.

Las respuestas continúan siendo incorrectas. El diagnóstico del día esperado
obtuvo **0/9 en ambos brazos y modelos**. La revisión de esas salidas por Codex
confirma errores visibles de lenguaje y atribución; no es una revisión humana
independiente ni una aceptación de conversación. Recuperar el texto correcto
no basta para que los pesos conversacionales sepan interpretarlo.

| Generador | Tiempo de 12 llamadas sin gestor | Con gestor |
| --- | ---: | ---: |
| Referencia `reference-wiki1` | 299,65 s | 15,01 s |
| Candidato de memoria | 250,33 s | 15,17 s |

Estos tiempos corresponden a entradas y respuestas distintas. La reserva evita
llenar y deslizar la ventana de generación, pero la comparación no acredita
una aceleración general ni igualdad de tokens generados. Se usaron un hilo,
temperatura cero, semilla 42 y un máximo de **24 tokens en ambos brazos**.
Ese límite acota la evaluación semántica. Un ensayo inicial de 64 tokens resultó
costoso por reconstrucciones completas de la caché; sus diez observaciones
parciales se conservan en [el registro de recursos](generation64-partial.json).
Se redujo el límite por tiempo de ejecución, sin ajustar los pesos ni la
selección mediante estas respuestas reservadas.

Para repetir con los mismos pesos y casos:

```bash
python examples/evaluate_context_integration.py --config configs/gnosis.json --output private/context-generation-reference.json
python examples/evaluate_context_integration.py --config configs/sovengi-memory-candidate.json --output private/context-generation-candidate.json
```

La entrega mejora la conservación de fuentes y su coste de recuperación. El
seguimiento de referencias cortas y la conversación competente siguen siendo
trabajo pendiente; no se presenta el selector como un generador nuevo.
