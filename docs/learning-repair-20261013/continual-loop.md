# Supervisor continuo local

`examples/continual_training.py` supervisa etapas nativas con límites finitos.
El bucle puede esperar nuevas etapas indefinidamente mientras el proceso y el
entorno sigan activos. No garantiza mejora, continuidad fuera del entorno ni
conversación competente. No cambia el modelo predeterminado ni promueve C7.

El supervisor inicial se inició el 13 de octubre de 2026 a las 04:52:32 UTC:
PID **10469**, nacimiento **1051690**. Adoptó el entrenamiento existente PID
**9540**, nacimiento **805298**, sin crear otro entrenamiento. En la observación
de adopción, el trabajador seguía `running` en el paso 3057. Estas cifras son una
observación histórica; `status` comprueba la situación actual con PID y nacimiento.

## Operación

```bash
python examples/continual_training.py start --config private/learning-repair-20261013/configs/continual.json
python examples/continual_training.py status --config private/learning-repair-20261013/configs/continual.json
python examples/continual_training.py stop --config private/learning-repair-20261013/configs/continual.json
```

`run` ejecuta el mismo supervisor en primer plano. `start` deja un proceso
independiente del terminal. El bloqueo exclusivo
`private/.continual-training.workspace.lock` permite un supervisor por workspace,
aunque se utilicen distintos directorios de estado. Los bloqueos nativos de cada
etapa siguen vigentes. `status` distingue proceso vivo de un estado grabado por
un proceso que ya murió; no presenta archivos antiguos como trabajo activo.

El supervisor se recargó, sin señalar al trabajador nativo, a las 04:57:01 UTC: PID **10516**, nacimiento **1078606**. La configuración activa usa `private/learning-repair-20261013/loop-inbox`; el recibo privado conserva los SHA anteriores y actuales.

El archivo de configuración es estático y queda ligado a su SHA-256 en el estado.
No se modifica para añadir etapas. Las nuevas etapas llegan por el inbox local.
Los caminos provienen de configuración del producto y permanecen dentro del
workspace; las respuestas del modelo y los mensajes de conversación no eligen
archivos, herramientas ni arquitectura.

## Configuración inicial

El archivo privado inicial contiene esta estructura. Sustituir los marcadores
por hashes reales; el archivo utilizado por el proceso ya tiene esos hashes.

```json
{
  "schemaVersion": 1,
  "workspaceRoot": "/workspace/UNIDAD",
  "stateDir": "private/learning-repair-20261013/continual",
  "inboxDir": "private/learning-repair-20261013/loop-inbox",
  "pollSeconds": 10,
  "maxCycles": null,
  "maxCrashRestartsPerStage": 2,
  "approvedAgents": ["project-agent-with-owner-standing-authorization"],
  "initialCycle": {
    "cycleId": "initial",
    "stages": [{
      "id": "foundation-15m",
      "kind": "base",
      "runDir": "private/learning-repair-20261013/runs/foundation-15m",
      "corpusPath": "private/learning-repair-20261013/corpus/causal-spanish-v1/causal-spanish.train-validation.jsonl",
      "corpusSha256": "5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17",
      "trainingConfigPath": "private/learning-repair-20261013/configs/foundation-training.json",
      "trainingConfigSha256": "<sha256>",
      "modelConfigPath": "private/learning-repair-20261013/configs/foundation-model.json",
      "modelConfigSha256": "<sha256>",
      "stopAfterStep": 12000,
      "invocationBudgetSeconds": 3600,
      "maxRSSMiB": 6144,
      "maxDiskMiB": 2048,
      "adoptExisting": true,
      "adoptPid": 9540,
      "adoptProcessBirth": "805298"
    }]
  },
  "collector": {"enabled": false}
}
```

Cada configuración nativa tiene `max_seconds <= 7200`, `max_steps <= 12000`,
`torch_threads <= 4`, objetivo `causal_lm` o `assistant_last_reply`,
`partition_policy: train_validation_only`, `data_loading: mmap` y todos los
parámetros entrenables. `stopAfterStep` debe coincidir con el límite persistente.
El runner nativo vigila hasta 6 GiB RSS y 2 GiB de archivos de la etapa mediante
paradas cooperativas; no son límites duros del sistema operativo.

Al terminar el presupuesto de una invocación de hasta una hora, se vuelve a llamar
`run_learning_stage.py --resume` sólo si queda presupuesto persistente y existe
un checkpoint coherente. Conserva Adam, scheduler, orden de datos y RNG. Una nueva
etapa desde pesos propios es otra operación: reutiliza artefacto y tokenizador,
y el entrenador crea un optimizador nuevo. No se amplían los límites anteriores
ni se reabren los presupuestos históricos cerrados.

Antes de adoptar o reanudar se comprueban corpus, configuración, fuente congelada,
particiones nativas y límites. Un PID vivo se espera incluso si `status.json`
ya dice `stopped` o `completed`: el runner puede seguir exportando. Sólo tras su
salida se consume el recibo de la invocación terminal y el snapshot seleccionado
por validación, con sus cuatro hashes y `modelId` exactos. Si murió después de
cerrar el entrenamiento pero antes de publicar el recibo, se permite una
recuperación de exportación, sin actualizar pesos ni reiniciar el entrenamiento.
Los fallos y las recuperaciones quedan registrados, sin inventar métricas de
memoria de una invocación anterior.

## Inbox y siguientes etapas

Un ciclo nuevo usa dos archivos:
`inbox/cycle-N.json` y `inbox/cycle-N.approval.json`. Este último es un **recibo
interno de procedencia bajo la autorización permanente del creador**, no una
solicitud interactiva de permiso ni una revisión humana independiente.

```json
{
  "approved": true,
  "manifestSha256": "<sha256 exacto de cycle-N.json>",
  "approvedBy": "project-agent-with-owner-standing-authorization",
  "approvedAtUtc": "<fecha UTC>",
  "standingOwnerAuthorization": true,
  "independentHumanReview": false
}
```

El manifiesto contiene `cycleId` y `stages`. Una etapa SFT puede usar
`parentStage: "initial:foundation-15m"`; el supervisor resuelve el snapshot
terminado y comprueba los cuatro hashes del recibo. No es necesario adivinar por
adelantado el nombre de su checkpoint. Para referirse a otra etapa del mismo ciclo
basta su identificador. Un padre fijo alternativo utiliza `parentModelPath` y
`parentFiles` con los cuatro hashes. Un modelo aleatorio utiliza `modelConfigPath`
y `modelConfigSha256`; no se combinan padre y modelo aleatorio.

Cada etapa nueva requiere `provenanceManifestPath` y
`provenanceManifestSha256`. El recibo de procedencia debe contener:

```json
{
  "corpusSha256": "<sha256 del corpus de la etapa>",
  "splitCounts": {"train": 100, "validation": 20, "test": 0},
  "testUsedForTraining": false,
  "contentSha256": "<hash de contenido sin metadatos>",
  "sources": [{"dataset": "<fuente autorizada>", "license": "<licencia real>", "receipt": "<procedencia>"}]
}
```

Los números del ejemplo no son mediciones del corpus real. El entrenador vuelve
a comprobar las particiones efectivas antes de empezar. Las fuentes Wiki/OASST
permitidas pueden prepararse y registrarse por agentes del proyecto dentro del
encargo, sin pedir otro prompt. El ciclo debe introducir contenido de corpus nuevo con su hash; la configuración
puede conservar su hash cuando la receta se reutiliza sobre datos nuevos; no repite indefinidamente una intervención fallida sobre los
mismos datos. Cambiar sólo metadatos no acredita contenido nuevo; el recibo del colector registra la ampliación efectiva. Un fallo nativo o de recursos conserva checkpoints y cierra ese
ciclo con rechazo, permitiendo esperar un manifiesto corregido nuevo. Sin nuevas
entradas, el estado es `waiting_for_data`, no `training`.

Puede configurarse un colector local opcional con `collector.enabled`, `argv`,
`sourceSha256` —caminos de sus scripts y hashes—, `timeoutSeconds` finito y
`everySeconds`. `{inbox_dir}` se sustituye por el inbox configurado. Se ejecuta
sólo durante la espera de datos, con PID/nacimiento, log y límite de duración.
El colector prepara procedencia y manifiestos; no decide el éxito del modelo ni
puede alterar el núcleo congelado de una etapa activa. Antes de lanzar un trabajador o colector se exige espacio libre del workspace
de al menos `maxDiskMiB + 512 MiB`; si falta, se registra `blocked_resources` y
se espera sin borrar historia. Una etapa SFT de retención puede reutilizar sus
datos si el mismo ciclo contiene una base con contenido nuevo, declarando
`retentionOnlyStage: true` y `novelDataParentStage`; no se presenta esa retención
como diversidad adicional.

La configuración inicial mantuvo el colector deshabilitado mientras se fijaban
código y procedencia. El 13 de octubre a las 05:37:05 UTC se activaron el
colector fijado y el hook de desarrollo mediante una recarga exclusiva del
supervisor: PID **12216**, nacimiento **1319028**, configuración SHA-256
`b251d50d0fe43868f29883c98b948f23b0a1d902c3013c0c36c05dd445797106`.
El [recibo de activación](continual-full-loop-activation.json) conserva las
identidades y hashes. El trainer 11176 continuó vivo y avanzó, sin recibir
señales. En esa observación no se inició ninguna descarga ni generación: ambas
esperan sus condiciones de cierre de SFT y espera de datos, respectivamente.
La corrección final del pin de colector se activó a las 05:39:12 UTC mediante
otra recarga exclusiva del supervisor: PID **12286**, nacimiento **1331693**,
configuración SHA-256
`b1b7dfb39d4744f8ffb2eafa981b2c60b3f0fd8b72bcd9977f4cceae58551bad`.
El [recibo del pin final](continual-final-collector-pin-activation.json) registra
que el trabajador 11176 siguió vivo, sin señales y con su mismo nacimiento.
El colector `examples/collect_learning_wiki.py` prepara ciclos base→SFT con
nuevas páginas fijadas de Wikipedia y SFT de retención declarada, sin promover
modelos ni afirmar mejora por pérdida.

## Estado, parada y evaluación

`state.json` conserva posición del ciclo, etapas terminadas, entradas consumidas,
identidades verificadas, recuperaciones, heartbeat y `conversationCertification`.
`journal.jsonl` registra adopciones, cambios de estado, lanzamientos y salidas
reales. Una salida cuyo código no se observó se llama `dead_exit_code_unknown`.
Los logs se guardan fuera del directorio nativo. No se borran candidatos ni
checkpoints para aparentar progreso.

La petición de parada del supervisor se guarda fuera de la etapa nativa. Si hay
trabajador vivo validado, publica una petición nativa completa de forma exclusiva
atómica y deja que cierre el siguiente límite coherente del optimizador. No
sobrescribe peticiones ajenas. Después deja de lanzar etapas. Para reactivar,
conservar/retirar expresamente la petición local de parada y ejecutar `start`;
una pausa nativa por una causa distinta de presupuesto no se reanuda sola.

La certificación predeterminada es `awaiting_review`. Menor NLL, más actualizaciones
u horas de proceso no la cambian. Un recibo de revisión aprobado internamente
puede declarar `accepted_basic_suite` sólo para el artefacto exacto, con:

- `modelId` y `artifactFiles` iguales al snapshot;
- `engine: ConversationEngine`, `actualGeneratedHistory: true`;
- `freshProspectiveSuite: true` y `candidateFrozenBeforeReservedOpening: true`;
- caminos y SHA-256 de `outputs`, `review`, `assessment`, `suiteFreeze` y
  `candidateFreeze`, más `suiteDir`;
- 40 turnos únicos, diez diálogos de cuatro turnos, historia real comprobada,
  candidato congelado antes de generar y revisión semántica identificada;
- al menos 36 completas, ocho diálogos completos y cero fallos críticos o técnicos.

Los campos se llaman `outputsPath`/`outputsSha256`, `reviewPath`/`reviewSha256`,
`assessmentPath`/`assessmentSha256`, `suiteFreezePath`/`suiteFreezeSha256` y
`candidateFreezePath`/`candidateFreezeSha256`. `reviewReceiptPath` y
`reviewApprovalPath` son opcionales en el ciclo; el archivo de aprobación liga
`receiptSha256` y un aprobador configurado. El supervisor recalcula los recuentos
desde los cuarenta juicios congelados y conserva el carácter de revisión de IA.
Nunca realiza una promoción automática. Una suite consumida no certifica otro
candidato. Las nuevas suites reservadas permanecen cerradas hasta fijar el
candidato; no se emplean como corpus, filtro ni objetivos de entrenamiento.

Se pospusieron las baterías de pruebas de software por indicación del creador.
Esta entrega se revisó estáticamente y la adopción del proceso real se comprobó
sin detenerlo. La primera reanudación automática se observó a las 05:11:32 UTC: el trabajador
9540 cerró por `invocation_time_budget` en el paso 4360 y 3600,90 s; el supervisor
11159 inició el trabajador 11176, nacimiento 1165743, con `--resume`. El nuevo
proceso conservó el paso y el tiempo acumulado y avanzó, con el mismo corpus y
configuración. El supervisor se había recargado sin señalar al trainer a las
05:09:42 UTC (PID11159, nacimiento1154726). El [recibo de esa reanudación](foundation-15m-first-resume.json)
identifica comando, checkpoint y procesos. El código nativo congelado carga Adam,
scheduler y RNG del checkpoint; no se ejecutó una comparación nueva de tensores.
Las transiciones de otras etapas todavía requieren observar sus eventos reales.

## Generación opcional al cerrar SFT

La configuración estática puede incluir `postStage`. Este hook sólo genera los
cuarenta turnos de desarrollo con `ConversationEngine` y la historia real. No
abre una suite reservada, rellena juicios semánticos ni acepta o promueve el
modelo. Después de la generación, la certificación sigue `awaiting_review`.

```json
{
  "postStage": {
    "enabled": true,
    "purpose": "development_generation",
    "torchThreads": 1,
    "timeoutSeconds": 300,
    "maxRSSMiB": 6144,
    "maxDiskMiB": 512,
    "suiteDir": "docs/learning-repair-20261013/development-suite",
    "argv": [
      "<mismo sys.executable local>",
      "examples/run_learning_evaluation.py",
      "--suite-dir", "docs/learning-repair-20261013/development-suite",
      "generate", "--model-dir", "{snapshot}",
      "--output-dir", "{output_dir}", "--torch-threads", "1"
    ],
    "sourceSha256": {
      "<script de generación>": "<sha256>",
      "<suite>/cases.json": "<sha256>",
      "<suite>/rubric.json": "<sha256>",
      "<suite>/freeze.json": "<sha256>"
    }
  }
}
```

El fragmento privado `configs/continual-post-stage-recipe.json` de esta misión
contiene las rutas y hashes reales. Activarlo exige una recarga explícita sólo
del supervisor y un recibo de cambio de configuración; no se señala al
entrenamiento nativo. El hook está deshabilitado cuando falta configuración;
la misión actual lo activó en la recarga de las 05:37:05 UTC, pero todavía no
había completado ni generado ninguna conversación con él.

Al cerrar una etapa SFT y verificar su snapshot, el supervisor registra una tarea
persistente. Genera antes de iniciar la siguiente etapa o el recolector. Fija un
hilo de Torch y variables OMP/MKL/OpenBLAS, conserva PID y nacimiento del hijo,
comando y hashes, y registra su salida real o desconocida tras una recarga. La
vigilancia cooperativa admite un presupuesto de hasta 300 segundos, RSS hasta
6 GiB y salida hasta 512 MiB; puede detectar el límite en el siguiente sondeo.
La muestra de `/proc/VmRSS` es memoria del hijo de generación observada durante
los sondeos, no el pico del kernel ni la memoria del entrenador. Si no hubo una
muestra, el recibo registra `null` y cero muestras. El runner público añade
`resource.getrusage(RUSAGE_SELF)`: el pico real del proceso de evaluación hasta
construir el informe, con unidad explícita. Incluye importaciones, carga del
modelo y generación; tampoco es memoria del entrenador.

Los archivos se guardan sin edición en
`stateDir/development-generations/<ciclo-etapa-modelo>/`, con un log separado y
un recibo `postStageResults` en el estado. Un directorio existente se conserva
para revisión; no se sobreescribe ni se repite automáticamente un fallo. Un
archivo de salidas presente puede registrar errores de generación y nunca
sustituye la revisión de los hechos, correcciones y petición actual.
