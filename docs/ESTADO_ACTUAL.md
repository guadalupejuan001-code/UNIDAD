# Estado actual de Unidad

## SOVENGI 0.9: entrenamiento de la coordinación, 13 de octubre de 2026

Se entrenó y conectó un selector propio de 18 pesos en 1.000 épocas, con
38 preferencias TRAIN y 15 VALIDATION de candidatos realmente generados. El
[informe](sovengi-coordination-20261013/README.md) conserva corpus ficticio nuevo,
pesos congelados y la evaluación reservada posterior. TRAIN pasó de 9/36 a
12/36 cumplimientos literales; VALIDATION quedó en 4/12. TEST quedó en 2/12,
igual que las reglas, con una ganancia y una regresión. En 7/12 casos ningún
candidato cumplía el criterio; estas cifras no acreditan conversación natural.

`configs/sovengi-coordination-candidate.json` ejecuta el selector entrenado en
SOVENGI, conservando la salida original y ambas decisiones en la traza. La
configuración habitual mantiene las reglas anteriores. La integración del
catálogo está fusionada mediante el [PR 21](https://github.com/guadalupejuan001-code/UNIDAD/pull/21),
con cuatro tareas CI aprobadas. Esta etapa de entrenamiento está cerrada;
no se reutilizó TEST para ajustar sus parámetros.

## SOVENGI 0.8: coordinación y conexiones, 13 de octubre de 2026

La configuración común coordina los 35 Transformers propios del
[catálogo](../configs/conversation-models.json). Cada petición programa hasta
cuatro generadores: dos anclas disponibles y dos puestos rotatorios, con un
mínimo de dos candidatos y un máximo de cuatro motores residentes. Conserva
la identidad del ganador y su texto original; los tokenizadores distintos se
comunican mediante texto y puntuaciones relativas dentro de cada juez.

La [entrega de coordinación](sovengi-ensemble-20261013/README.md) conserva
identidades, reconstrucción exacta del delta, pruebas y mediciones. La
[auditoría de conexiones](sovengi-connectivity-20261013/audit.json) distingue
los componentes y su trabajo real: contexto y memoria, diagnóstico RNN,
pronóstico MLP con observaciones explícitas, autonomía y etapas GNOSIS. Torch
con caché sigue siendo el decodificador predeterminado; NERVA se selecciona
desde la configuración del anfitrión.

Esta entrega integra la ejecución y no entrena nuevos pesos. La calidad
conversacional y la aceptación general siguen sin acreditarse. Los cortes de
entrenamiento y resultados anteriores que siguen se conservan como historia.

## Historial: integración y entrenamiento cerrado el 13 de octubre

El código verificado en `main` `2d989a5` reúne SOVENGI, memoria recuperada,
herramientas, API y NERVA conversacional. `configs/gnosis.json` carga el modelo
propio `reference-wiki1` de 15.769.344 parámetros y contexto 2.048, con Torch y
caché. C7 permanece como referencia histórica.

La [auditoría y etapa nueva](sovengi-memory-20261013/README.md) comprueban las
ramas y documentan el entrenamiento de memoria y correcciones. La ejecución
[38433132860](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38433132860)
terminó: 27 actualizaciones, 921,32 segundos dentro del entrenador y parada por
presupuesto de tiempo. Su control de desarrollo pasó de 1/25 a 22/25 respuestas
exactas; las 20 consultas de memoria fueron correctas. La revisión conserva
respuestas aún deficientes y no equivale al test de aceptación general.

Los pesos nuevos están en `models/sovengi-memory-20261013/candidate-best`, y
`configs/sovengi-memory-candidate.json` los conecta al mismo SOVENGI para prueba
explícita. La referencia predeterminada y sus hashes se conservan. Esta corrida
está cerrada; no se anuncia entrenamiento continuo ni promoción automática.

Las observaciones de PID, colas y entrenamientos que siguen tienen la fecha de
su encabezado. No acreditan que esos procesos sigan vivos en otro entorno.
El [estado estructurado](unidad-current-status.json) identifica la configuración
actual y conserva el estado anterior como evidencia histórica.


## Integración conversacional de NERVA, 13 de octubre de 2026

La [entrega y medición](nerva-conversation-20261013/README.md) conecta NERVA a
la selección de tokens del Transformer mediante el motor compartido. Añade
caché por respuesta, controles de generación neutrales por defecto y una ruta
sin caché para comparación. CLI, Brain/API y GNOSIS pueden seleccionar NERVA;
el anfitrión fija esa elección, no el texto del diálogo.

Con C7, diez consultas conocidas y tres repeticiones, las medianas fueron
12,72 s sin caché, 4,37 s con caché Torch y 27,13 s con caché NERVA. Los tokens
y los artefactos coincidieron. Se adopta Torch con caché como opción rápida;
NERVA permanece disponible y comprobado. Esta medición corresponde a C7 y no
evalúa semánticamente ni promueve el candidato de patrones de 15,77 millones.
La intervención conservó pesos, datos, checkpoints y procesos de entrenamiento.
Las observaciones históricas de otros procesos mantienen sus fechas.

## Descargas completas y entrenamiento activo, 13 de octubre, 15:18:20 UTC

La [recuperación del lote pendiente](learning-repair-20261013/download-recovery/README.md)
completó 58 páginas, reutilizando 28 de caché. Se incorporaron 2.553 registros
nuevos de 2.545 artículos de Wikipedia en español, más 2.299 registros conocidos
de repaso. El corpus tiene 4.325 TRAIN, 527 VALIDATION y ningún TEST.

El supervisor 22557 admitió `wiki-cycle-0003`; el trabajador 33674, nacimiento
`4777437`, había ejecutado 251 actualizaciones causales en este corte. Continúa
desde pesos propios, con corpus y configuraciones fijados y presupuestos finitos.
No se detuvo ni reinició la corrida previa para recuperar las descargas.
Se entregan el corpus distribuible, su procedencia y un transporte optativo con
reintentos limitados. El supervisor vivo conserva su recolector original.

Las observaciones siguientes se conservan como historia, incluido el estado de
espera causado por HTTP 502. La descarga está completa; entrenamiento y evaluación
posteriores no se presentan como terminados ni como conversación aceptada.

## Historial: aprendizaje de patrones, 13 de octubre, 14:25 UTC

El [control de relaciones conversacionales](learning-repair-20261013/pattern-relations-v2/README.md)
integró datos contrastantes, evaluación real y una etapa en la cola existente.
Ejecutó 512 actualizaciones sobre los pesos propios de 15.769.344 parámetros
en 72,283 segundos. La validación seleccionó BEST 128, `af515755…`.
El padre, el candidato, el tokenizador y el checkpoint nativo BEST se entregan
con hashes, historial y respuestas antes/después.

En doce prefijos TRAIN, la revisión de IA pasó de cero a once completas.
En doce de desarrollo pasó de cero a una completa y una parcial; las
coincidencias exactas siguieron en cero. Los dos diálogos libres fallaron.
El control demuestra aprendizaje de ejemplos conocidos, sin cumplir el
criterio de generalización ni la aceptación conversacional. La reserva final
sigue cerrada y C7 permanece como referencia anterior propia.

En este corte el supervisor 22557 sigue vivo, en `waiting_for_data` después
de cerrar la etapa; no hay entrenador nativo activo. La etapa diagnóstica no
se convirtió en padre automático del colector. Este estado es una observación
fechada, no una promesa de ejecución futura fuera del entorno.

## Historial: avance posterior a PR11, 13:21 UTC

La descripción vigente está en [qué tenemos en Unidad](QUE_TENEMOS.md), el
[README](../README.md) y el [recibo posterior a PR11](learning-repair-20261013/progress-20261013-1304.md).
Las secciones de las 06:44 y anteriores que siguen se conservan como historia;
sus PID, colas y estados no representan la situación actual.

El código, los datos y los cinco artefactos BEST completos de PR11 se fusionaron
en `a948c0fd358b250c7137978e3077ac70cdc51d83`, con cuatro checks aprobados.
Después cerraron Tatoeba (2.048 + 1.024 actualizaciones) y el primer lote
automático de Wikipedia (1.772 + 1.024). El supervisor sigue vivo; el corte
actual registra espera de nuevos datos. Sus checkpoints posteriores son locales.

La última comparación revisada, tras Tatoeba, obtuvo 0/40 respuestas completas
y 0/10 diálogos en ambos juicios de IA, con nueve/diez críticos y cinco técnicos.
El anterior resultado de 1/40 se conserva, sin presentar este retroceso como
mejora. La generación más reciente se revisa por separado. C7 sigue como
referencia; conversación y planificación fiables permanecen pendientes.

## Historial de observaciones

## SFT en ejecución y PR8 integrado, 13 de octubre, 06:44 UTC

El [nuevo recibo](learning-repair-20261013/observation-20261013-064409-sft-active-pr8-merged.json)
registra **2.200 actualizaciones reales** de `conversation-15m` y **1.626,786
segundos** nativos acumulados. El entrenador **13379**, nacimiento **1558057**,
y el supervisor **13378**, nacimiento **1556040**, siguen vivos. En el
checkpoint del paso **2.048**, la NLL SFT validada es **1,735812**, frente a
**2,738114** al inicio de este ajuste. Esta métrica supervisa la última
respuesta y EOS; no equivale a conversación competente ni se compara
directamente con la NLL causal de la etapa anterior. Todos los 15.179.520
parámetros siguen entrenables, con corpus y tokenizador fijados.

[PR8](https://github.com/guadalupejuan001-code/UNIDAD/pull/8) se fusionó sin
borrador a las **06:42:44 UTC**. Su commit de trabajo es
`8a7dc3f43a5797fe81e0873ecbdd881c8e8a619f`; el merge en `main` es
`34e5fd0c772305acdb18616c684c7c4cf7450cac`. Incluye la reparación del supervisor,
los pesos y BEST/LAST finales de la base y las nuevas fuentes públicas.
La CI de ese commit terminó con éxito en
[PR](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38356784618)
y [push](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38356685472).
La [CI del merge](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38358257321)
seguía en ejecución al leer este recibo. No se ejecutó otra batería local.

La mezcla `wide-spanish-v1` continúa **en cola**, con padre en el SFT actual:
4.084 TRAIN, 621 VALIDATION y cero TEST. Los 2.406 registros nuevos reúnen
1.470.331 palabras entre TRAIN y VALIDATION; sus 2.299 anclas de retención
se cuentan aparte. Que los archivos estén en `main` no significa que esta
etapa haya entrenado. El colector y la generación DEV al cierre están
habilitados, pero aún no se ejecutaron. No se generaron conversaciones de
esta etapa ni se abrió la reserva final; C7 se conserva como referencia y
la aceptación sigue pendiente. Las observaciones anteriores permanecen intactas.

## Cierre de base y SFT activo, 13 de octubre, 06:23 UTC

La base propia `foundation-15m` terminó en **8.914 updates**, con **22.906.669
posiciones objetivo** procesadas en 71.297 ventanas. La NLL causal validada
final fue **2,981874**. La parada `total_time_budget` tenía umbral de 7.200
segundos; el contador final registra **7.337,688 segundos** incluyendo
validación y guardado de cierre. Es un límite cooperativo, no un corte exacto
de tiempo de pared. El [recibo final](learning-repair-20261013/foundation-15m-final-delivery.json),
la [contabilidad](learning-repair-20261013/foundation-15m-final-target-exposure.json),
los [pesos](../models/learning-repair-20261013/foundation-15m/README.md) y
[BEST/LAST](../checkpoints/learning-repair-20261013/foundation-15m/final/README.md)
se preservan con hashes; la entrega final está local para una subida posterior.

SFT ya entrena desde el checkpoint propio: **512 updates y 402,289 segundos**
en el [nuevo recibo temporal](learning-repair-20261013/observation-20261013-062351-base-complete-sft-active.json).
El trabajador **13379**, nacimiento **1558057**, y supervisor **13378**,
nacimiento **1556040**, estaban vivos. Objetivo `assistant_last_reply`, todos
los **15.179.520 parámetros** entrenables y corpus 18.588 TRAIN/1.704 VALIDATION,
sin TEST. No se generaron ni evaluaron conversaciones de esta etapa; C7 sigue
como referencia y `conversationCertification` permanece `awaiting_review`.

Un fallo real del supervisor tras cerrar la base bloqueó la transición:
`Supervisor.event()` recibía dos valores para `kind`. Se corrigió el nombre
posicional, una razón operacional obsoleta y la identidad equivalente de
`python`/`python3`. Se reemplazó sólo el supervisor bloqueado, sin señalar un
entrenador ni reiniciar la base. El
[recibo de reparación](learning-repair-20261013/continual-stage-transition-repair.json)
conserva fallo, cambios y arranque real de SFT. El núcleo ML quedó congelado.

[PR7](https://github.com/guadalupejuan001-code/UNIDAD/pull/7) está fusionado y sin
borrador: commit de trabajo `033148c6d0d09feefd6127f718b1dd4149cbad4c`,
merge en `main` `32d08dacb070e349e8bbf3213133cb695d3e5f19`. GitHub Actions
terminó con éxito en [trabajo](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38354851309)
y [main](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38355571276).
Esos resultados corresponden a PR7; el parche de transición y los archivos
finales nuevos siguen locales en esta observación. No se ejecutó otra batería
local para este recibo ni se atribuye calidad conversacional a CI.

La [ampliación `wide-spanish-v1`](../examples/learning-repair-20261013/wide-spanish-v1/README.md)
ya está en cola con padre `conversation-v1:conversation-15m`: 4.084 TRAIN,
621 VALIDATION y cero TEST; 2.406 registros nuevos, 2.299 de retención y
1.470.331 palabras de las fuentes nuevas entre TRAIN/VALIDATION. Sigue con
adaptación causal y SFT de retención finitos. No está entrenada todavía.
El colector y la generación DEV posterior a SFT están habilitados, pero no
se ejecutaron en esta lectura; la reserva final permanece cerrada. El bucle
depende del entorno vivo y puede esperar datos o recursos, sin garantizar
mejora indefinida. Los recibos de las 05:01 y 05:26 se conservan intactos.

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
