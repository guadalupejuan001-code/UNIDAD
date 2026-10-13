# Referencia al BEST final del tratamiento LATAM v4

BEST permanece en el paso 1.024 tras el cierre en 4.096, con NLL de validación
1,7603046832825393. [artifact-reference.json](artifact-reference.json) apunta al
[artefacto completo existente](../intermediate-step001024/), cuyos pesos,
configuración y tokenizador son idénticos a la selección final.

Este directorio de referencia no se carga directamente con el motor. Debe
resolverse `artifactDirectory` y cargarse ese artefacto completo. El
[manifiesto final](manifest.final-selection.json) conserva los bytes originales
de la selección por validación; el manifiesto intermedio conserva su selección
explícita anterior. No se duplicaron pesos ni se alteró la entrega intermedia.

La [entrega](../../../../docs/learning-repair-20261013/natural-latam-v4-treatment-final-delivery.md)
identifica exposición y recuperación. El candidato mantiene
`productionAccepted=false`.
