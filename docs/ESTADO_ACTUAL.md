# Estado actual de Unidad

Comprobado el **13 de octubre de 2026**. Se integra el código compatible de las
ramas y PR1–6, se entregan seis modelos propios entrenados y se actualiza el
README sin entregas en borrador. **La conversación fiable continúa pendiente**:
ningún candidato cumple la aceptación. C7 conserva su identidad y cuatro
archivos originales. No se presenta un perfil de mayor tamaño como entrenado.

## Resultado de aprendizaje

Las seis corridas completaron **2.560 pasos de Adam**, con todos los parámetros
entrenables: 76/76 tensores cambiaron en los brazos V7 y 80/80 en los brazos C7.
Se compararon exposición humana, historia ideal/generada y diálogos variados
originales de cuatro turnos. Disminuyó la NLL, pero no aumentaron de forma neta
las respuestas completas de desarrollo. Las intervenciones se detuvieron.

El snapshot de historia generada384 se fijó por identidad y cuatro hashes antes
de abrir otra prueba reservada. Se ejecutaron diez diálogos de cuatro turnos,
160 tokens, temperatura cero, semilla42 y el mismo `ConversationEngine`, usando
sus respuestas reales anteriores como historia.

| Modelo / revisión primaria de IA del proyecto | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C7 conservado | 0/40 | 4 | 36 | 0/10 | 2 | 0 |
| Historia generada384 | 0/40 | 1 | 39 | 0/10 | 2 | 0 |

No alcanzan **36/40 completas, 8/10 diálogos correctos y cero críticas/técnicas**.
No hubo vacías, recortes ni límites de tokens alcanzados en los 80 turnos. Son
revisiones de IA del proyecto, no humanas independientes. Se conservan sus
respuestas, rúbricas y discrepancias; no se entrenó ni cambió la selección por
la prueba final. Las baterías históricas diferentes conservan sus métricas y
no se mezclan con esta comparación.

El [informe de integración](conversation-integration-20261013/README.md) y la
[comparación nueva](conversation-integration-20261013/evaluation/reserved-comparison.json)
identifican casos, generación, modelos y hashes. El bloqueo observado es de
generalización a peticiones nuevas, recuperación de hechos y correcciones;
la base lingüística insuficiente sigue siendo una hipótesis, no una causa
completa aislada. No se demostró calidad competitiva.

## Código integrado y comprobaciones

PR1/3 aportan diagnóstico/evaluador y documentos históricos; PR2, proyección
final al generar; PR4, exportador ONNX con controles de publicación; PR5/6,
GNOSIS, NERVA y datos mmap. Se conservan mejoras posteriores y cambios ajenos,
incluido el README de `7990c4562852fda6ca366c058aacb1f781dda551`.
La [matriz de ramas](conversation-integration-20261013/branch-reconciliation.json)
identifica las ocho ramas originales y 324 documentos importados byte por byte.
El parche `source-pointer-v2.patch` no se aplica ni se promueven modelos fallidos.

Aprobaron **494 pruebas y 12 ejemplos** con conversación y ONNX instalados. El
fallo histórico `stale` frente a `planned` no se reprodujo: la prueba intacta
comprobó lanzamiento vivo, parada con checkpoint y corpus inválido fallido.
`stale` conserva su significado de trabajador que ya no vive. La proyección
final conservó cuatro respuestas C7; ONNX también reprodujo cuatro generaciones
greedy en CPU Linux. No se probó un teléfono físico. Las pruebas técnicas y las
mediciones de velocidad/memoria no prueban comprensión.

El control pequeño histórico ya mostró 16/16 respuestas TRAIN exactas a 64
pasos desde C7 y pesos aleatorios comparables, con mala transferencia de
formulación. Se verifica y conserva su evidencia original sin repetirla o
presentarla como un entrenamiento nuevo. Los
[controles de mecanismo](conversation-integration-20261013/learning-mechanism-checks.json)
y [logs](conversation-integration-20261013/software/verification.json) distinguen
memorización, funcionamiento y generalización.

## Entrega reproducible

- [Seis snapshots propios](../models/conversation-integration-20261013/README.md): pesos, configuración, tokenizador, manifiesto e identidad exacta.
- [Checkpoints BEST y ENDPOINT](../checkpoints/conversation-integration-20261013/README.md): historia generada y ambos ajustes naturales; Adam/programador/RNG validados y restaurados en directorios nuevos sin entrenar.
- [Corpus y atribuciones](conversation-integration-20261013/data/README.md): OASST público fijado, familias separadas, datos ficticios con autoría IA explícita, sin conversaciones privadas del creador.
- [Diseños y estados](conversation-integration-20261013/training/README.md): configuraciones, logs, límites, pasos, tiempos y cambios reales de tensores.
- [Fuentes metodológicas originales](conversation-integration-20261013/methods-original-sources.md): técnicas consideradas y límites de su aplicación local.

Para cargar exactamente el candidato evaluado, con dependencias de conversación:

```sh
python -m ia_propia.conversation converse \
  --model models/conversation-integration-20261013/history-generated \
  --max-new-tokens 160 --temperature 0 --seed 42
```

Su identidad es `unidad-conversation-be6414b4187c33e27aef54e67352aaa78393afb303f7b8daaf8c0eb7c2a558a2`.
El [recibo de carga](conversation-integration-20261013/software/published-model-cli.json)
comprueba los archivos y la consola. El candidato es experimental y rechazado.

## Historia y procesos

La [rama de artefactos](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012)
conserva C7 y los candidatos original128/relacional128 con sus checkpoints.
Las ramas históricas y [el estado anterior](historial-estado-20261013-0036.md)
conservan la disponibilidad exacta de paquetes y sus resultados; no se presenta
como restaurado un ZIP ausente. El
[estado estructurado](unidad-current-status.json) identifica la entrega nueva y
el snapshot anterior.

Las nuevas corridas autorizadas terminaron. En este entorno no hay entrenamiento
ni generación de esta etapa activos. Guardar Adam/RNG no inicia una continuación.
Los presupuestos históricos 24+2 horas permanecen cerrados; no se contrataron
servicios ni se usó GPU. No se inspeccionaron procesos de otros entornos.
