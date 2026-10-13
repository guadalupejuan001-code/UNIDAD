# Estado actual de Unidad

## Avance del 13 de octubre, 05:26 UTC

La base propia continúa tras una **reanudación real**: la primera invocación
cerró en step 4.360, a los 3.600,901 segundos nativos, por
`invocation_time_budget`. El supervisor lanzó `--resume` a las 05:11:32 UTC
con PID 11176 y nacimiento 1165743. A las 05:26 UTC se registran 5.460 updates
y 4.468,758 segundos acumulados; ese trabajador y el supervisor 11159,
nacimiento 1154726, estaban vivos. La NLL validada es 3,2752 en step 5.120.
El [recibo de reanudación](learning-repair-20261013/foundation-15m-first-resume.json)
confirma fuente/entradas intactas y que no se reinició el optimizador. El
[recibo posterior](learning-repair-20261013/observation-20261013-052609-resume-sft-queued.json)
se añade sin reescribir la observación de las 05:01 UTC que sigue abajo.

La etapa `conversation-v1:conversation-15m` está **encolada**, con padre
`initial:foundation-15m`; todavía no se ejecutó SFT. El corpus fijado contiene
18.588 TRAIN, 1.704 VALIDATION y cero TEST; 7.676 TRAIN humanos públicos
(41,295 %). Se excluyeron siete objetivos erróneos TRAIN y uno VALIDATION sin
alterar su historia correctiva, y se retiraron cinco diálogos exactos repetidos.
No se atribuye conversación competente a ninguna pérdida ni preparación.

Se prepararon [entradas distribuibles](learning-repair-20261013/distributed-inputs.json),
[plantilla pública de dos etapas](../examples/learning-repair-20261013/continual-template.json)
y un [checkpoint intermedio](learning-repair-20261013/foundation-15m-step-3072-delivery.json)
de step 3.072 en nueve partes con hashes; su restauración y subida aún no
figuran como ejecutadas. El trabajo nuevo continúa local y pendiente de
integración; `main` conserva la entrega anterior. El colector permanecía
deshabilitado en esta lectura. No se añadieron pruebas de software ni se abrió
la evaluación conversacional final.

## Observación anterior conservada, 05:01 UTC

Observación del **13 de octubre de 2026, 05:01 UTC**. Una nueva base propia de
15.179.520 parámetros está entrenando desde pesos aleatorios: el estado nativo
registra 3.610 updates, PID 9540 y nacimiento 805298 comprobados. El supervisor
10516, nacimiento 1078606, está vivo y adoptó esa misma corrida. La etapa SFT
posterior se prepara; no se ha evaluado conversación de esta nueva base.
El [recibo temporal](learning-repair-20261013/current-observation.json) conserva
la lectura y la [continuación](learning-repair-20261013/current-delivery.md)
describe hipótesis, datos y límites. Los archivos de estado cambian y los PID
pueden terminar; esta fecha no garantiza que sigan activos al leer el documento.

PR1–6 están fusionados y sin borradores; `main` remoto está en
`242aa3e57368352f72bea6704833f70be376b84c`. La continuación nueva se trabaja en
`work/learning-repair-20261013`, pendiente de un recibo posterior de integración.
**La conversación fiable continúa pendiente**: ningún candidato anterior cumple
la aceptación. C7 conserva su identidad y cuatro archivos originales.

## Continuación de aprendizaje

El corpus causal fijado contiene 10.735 documentos TRAIN y 1.351 VALIDATION,
con texto público de Wikipedia en español, OASST1 y OASST2. TEST se mantiene
separado y la caché nativa contiene cero ejemplos TEST. BPE de 2.048 tokens,
ocho capas y contexto de 512; todos los parámetros se actualizan, con batch
lógico ocho, microbatch cuatro, mmap y BF16 en CPU. El tokenizador utiliza
sólo TRAIN y no se incorporaron mensajes privados del creador.

La validación causal sobre las mismas 687.867 posiciones registra NLL 7,7110
en step 0 y 3,7734 en step 3.072. Hay aprendizaje de predicción de texto en esa
métrica; no demuestra continuidad de diálogo, correcciones o generalización.
Se prepara SFT con diálogos humanos públicos y ficción sintética identificada,
sin abrir la evaluación conversacional final ni usarla para elaborar datos.

La etapa tiene un máximo persistente de 7.200 segundos o 12.000 updates y
techos cooperativos de 6 GiB RSS y 2 GiB en el directorio nativo. Las invocaciones
de hasta una hora se reanudan dentro de ese presupuesto, preservando Adam,
programador, orden y RNG. El supervisor puede esperar nuevas entradas mientras
el entorno siga activo; el colector está deshabilitado en esta observación.
No se afirma que las transiciones futuras ya hayan ocurrido.

## Resultado anterior conservado

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
[estado estructurado anterior](unidad-current-status.json) identifica aquella
entrega y su snapshot; el [recibo temporal nuevo](learning-repair-20261013/current-observation.json)
identifica la continuación activa observada.

Las seis corridas de la integración anterior terminaron. La continuación propia
de 15,18 M está activa en la observación de 05:01 UTC; sus archivos se conservan
en rutas privadas mientras se ejecuta. Guardar Adam/RNG no inicia por sí solo
otra corrida. Los presupuestos históricos 24+2 horas permanecen cerrados;
esta etapa nueva responde a la autorización posterior. No se contrataron
servicios ni se usó GPU. No se inspeccionaron procesos de otros entornos.
