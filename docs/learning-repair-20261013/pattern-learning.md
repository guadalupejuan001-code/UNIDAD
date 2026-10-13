# Patrones conversacionales: mecanismo y aprendizaje

Esta intervención responde a la petición de comprobar el reconocimiento de
patrones e integrarlo donde falte. UNIDAD ya dispone de atención causal,
embeddings de tokens y posiciones y capas neuronales GELU. Estos componentes
aprenden asociaciones al actualizar sus propios parámetros. Su existencia no
demuestra que el modelo distinga personas, seleccione un hecho o aplique una
corrección correctamente.

El fallo observado que motiva el experimento es concreto: en la evaluación de
desarrollo posterior a Tatoeba, `library_day_correction` seguía hablando del
viernes después de que el usuario corrigiera el día a jueves. El contexto tenía
324 tokens y no se descartaron tokens. La comparación revisada obtuvo 0/40
respuestas completas. No se atribuye este fallo a la ausencia de atención ni se
declara una causa única sin medirla.

## Hipótesis fijada antes de entrenar

Un ajuste pequeño con relaciones contrastantes puede enseñar al modelo vigente
a depender del hecho relevante del diálogo, en vez de repetir una respuesta
frecuente. Los pares mantienen la misma pregunta y cambian el contexto que
determina su respuesta. El currículo cubre hechos por persona, correcciones,
negación, información ausente y seguimiento de la petición actual.

Se conserva la arquitectura, el tokenizador, el motor de conversación y la
política de supervisar sólo la última respuesta y EOS. Los errores previos del
asistente sirven como contexto de reparación; no se convierten en objetivos
positivos. Las respuestas de entrenamiento son datos ficticios declarados. La
inferencia sigue generando texto con los pesos propios: no recibe un solucionador
de reglas ni respuestas de plantilla.

El padre fijado es el checkpoint propio cerrado posterior al primer lote
automático de Wikipedia: `c0ee5f24b2424eb652f3ca9bda5fcb8841d703b4f8f5efc4db23427a8ed123c4`.
Tiene 15.769.344 parámetros y contexto de 2.048 tokens. Se compara el padre sin
este ajuste con el candidato seleccionado por validación nativa. Es un control
pequeño de aprendizaje, no una comparación con un brazo de otro entrenamiento
de igual presupuesto ni una medida de calidad general.

## Límites y criterio del control

La etapa tiene como máximo 512 actualizaciones, 600 segundos cooperativos,
cuatro hilos CPU, 6 GiB de RAM y 1 GiB de archivos de corrida. El supervisor
existente la recibe por su cola y espera el cierre de la corrida activa. No se
reinician presupuestos cerrados ni se modifica la configuración de esa corrida.
La validación, el guardado y la exportación pueden añadir tiempo al límite
cooperativo; se conserva el tiempo real observado.

El control de generación usa hasta doce prefijos TRAIN y doce de desarrollo,
además de dos diálogos de desarrollo con historia realmente generada. Antes de
la corrida se fija este objetivo diagnóstico: al menos diez de doce coincidencias
TRAIN con EOS, ocho de doce en desarrollo, ambos diálogos completos, mejora
frente al padre y ausencia de errores técnicos o pérdida de contexto. Las
coincidencias exactas son una medida estrecha del formato de esta tarea: una
paráfrasis correcta puede no coincidir. La probabilidad de una respuesta ideal,
si se mide, es un diagnóstico adicional y no sustituye generación libre.

Los prefijos ideales se identifican aparte. Los nombres, valores, plantillas y
familias de desarrollo se mantienen separados de entrenamiento; los pares de
una familia no se reparten entre particiones. Desarrollo no es la reserva final.
Esta intervención no abre la evaluación reservada ni acredita el criterio de
36/40 respuestas, ocho de diez diálogos y cero fallos críticos o técnicos.

El tipo de etapa es `pattern_diagnostic`. Sus pesos no pasan automáticamente a
ser el padre del colector de español, y no se promueven por bajar la pérdida.
Se conserva el resultado aunque el control memorice y falle la generalización.

## Resultado

Pendiente de ejecución y comparación. Preparar este documento o un corpus no
actualiza los pesos. Los recibos de entrenamiento y las respuestas reales se
añadirán cuando existan.
