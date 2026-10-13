# Diagnóstico del aprendizaje conversacional

La inspección y las pruebas técnicas no encontraron un defecto en el desplazamiento next-token, la alternancia de roles ni la supervisión exclusiva de la última respuesta/EOS. El contrato y `ConversationEngine` conservaron el historial realmente generado, y los corpus nuevos cabían completos en el contexto. Esto describe lo comprobado; no demuestra que no pueda existir otro defecto. Las verificaciones de cada brazo conservan el contexto máximo, corpus, código y estado efectivo.

## Evidencia de concentración del entrenamiento

El [manifiesto original C7](../../../models/conversation-reference-c7/manifest.json) registra **61,600 ventanas de escenas sintéticas**, con participación esperada **74.00%**. La fuente humana OASST tiene sólo **34 ventanas** y **1.00%** de participación. En [V7](../../../models/conversation-language-v7/manifest.json), OASST conserva **14,310 ventanas**, con **20.06%**. Son distribuciones de muestreo registradas, no recuentos de respuestas correctas ni una comparación causal controlada entre ambos linajes.

Es compatible con un sesgo hacia marcos de respuesta repetidos. La base española ancestral registra NLL de lenguaje **4,77545**, y las respuestas libres conocidas contienen tautologías, frases ajenas a la pregunta y errores al escoger el hecho relevante. La hipótesis de una base lingüística y composicional insuficiente sigue abierta. No se deduce de esto una cantidad necesaria de parámetros, tokens u horas ni una garantía de resolverla aumentando el tamaño.

## Controles nuevos

Se completaron tres contrastes acotados: español humano breve desde V7 y C7; historia ideal frente a historia realmente generada por C7 sobre iguales usuarios/objetivos; y diálogos de cuatro intercambios redactados por tres asistentes de IA del proyecto, con anclas humanas idénticas entre los dos linajes. Los autores de los diálogos no son evaluadores humanos independientes. Las etapas conservaron la procedencia propia desde pesos aleatorios, la partición familiar, todos los parámetros entrenables y el tokenizador propio.

| Brazo | NLL inicial | NLL seleccionada | Paso seleccionado |
| --- | ---: | ---: | ---: |
| v7 | 4.799321 | 4.635460 | 256 |
| c7 | 5.508405 | 4.490238 | 256 |
| history-ideal | 0.482787 | 0.061610 | 512 |
| history-generated | 0.482787 | 0.064625 | 384 |
| natural-v7 | 4.995627 | 4.718165 | 128 |
| natural-c7 | 5.162015 | 4.150266 | 128 |

Las pérdidas anteriores corresponden a distintos corpus/reducciones y no deben ordenarse como una escala única de capacidad. En el contraste de historia, las dos validaciones usan la misma historia ideal conservada; la exposición a errores reales cambia sólo en TRAIN. El corpus generó 576 respuestas previas de C7 y cambió el contexto de 368 prefijos, conservando las respuestas finales correctas. Esto prueba el tratamiento aplicado, no que la exposición a errores sea la causa principal.

La disminución de NLL no acreditó generalización conversacional en las pruebas libres de desarrollo descritas por el informe de integración. Por eso estos controles conservan pesos y evidencia, pero no se promocionan por pérdida ni sustituyen la aceptación de **36/40 respuestas completas, 8/10 diálogos completos y cero fallos críticos o técnicos**. La reserva final se gestiona después de congelar el candidato y sus respuestas se publican en el informe correspondiente; este diagnóstico no la usa para modificar datos o pesos.

Los puntos comprobados permiten seguir integrando etapas y reproducir artefactos/estados. No acreditan conversación básica, calidad competitiva ni programación autónoma.
