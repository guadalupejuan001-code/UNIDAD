# Continuación de aprendizaje propio

Observación fijada el **13 de octubre de 2026 a las 05:01 UTC**. El modelo
`foundation-15m` está entrenando; el estado nativo registra **3.610 updates**
completados y 2.960,685 segundos acumulados. El proceso PID **9540**, nacimiento
**805298**, y el supervisor PID **10516**, nacimiento **1078606**, existían con
esas identidades. El [recibo temporal](current-observation.json) recoge los
valores, las validaciones y el estado de GitHub. Es una lectura histórica;
para saber si el proceso sigue vivo debe consultarse su estado actual.

La conversación competente continúa pendiente. Este documento no acredita
SFT ejecutado, respuestas nuevas ni un candidato aceptado de esta continuación.
C7, los seis ajustes rechazados y las evaluaciones anteriores conservan sus
pesos, identidades y resultados. No se modifican respuestas reales ni se llama
a otro modelo para sustituir la inferencia de Unidad.

## Hipótesis e intervención

Las intervenciones cortas desde C7/V7 anteriores no mejoraron de forma neta
las respuestas completas. La hipótesis de trabajo es que una base lingüística
con más exposición y capacidad, seguida de ajuste supervisado conversacional,
puede aprender mejor español y trasladarlo a peticiones nuevas. La hipótesis
no establece una causa única de los fallos anteriores.

Se entrena otra base propia desde pesos aleatorios, aprovechando el entrenador,
contrato, mmap y generación existentes. Cambian corpus, vocabulario, capacidad
y programación de tasa: esta intervención combina esas variables y no permite
atribuir una mejora futura a una de ellas por separado. El resultado negativo
anterior se conserva en el [informe de integración](../conversation-integration-20261013/README.md).

La base tiene **15.179.520 parámetros**, ocho capas de dimensión 384, ocho
cabezas, contexto de 512 y BPE de 2.048 tokens ajustado sólo con TRAIN.
Todos los parámetros son entrenables; no se limita el ajuste a una cabeza de
copia. No hay pesos preentrenados externos. El backend es PyTorch en CPU;
esta etapa no migra el Transformer a NERVA ni atribuye GPU al entorno.

Se mantiene la secuencia base causal → SFT de diálogos. El método es
supervisión explícita con procedencia y particiones; no se ha añadido PPO,
GRPO ni DPO. Las [publicaciones originales consultadas](../conversation-integration-20261013/methods-original-sources.md)
fundamentan la separación entre base lingüística, ajuste de instrucciones y
evaluación, sin copiar recetas completas ni hiperparámetros de otro producto.

## Datos nuevos y filtros

El corpus causal contiene **10.735 documentos TRAIN**, **1.351 VALIDATION** y
**1.405 TEST** separados. Sólo TRAIN y VALIDATION se suministraron al entrenador.
El archivo utilizado ocupa 27.132.589 bytes y tiene SHA-256
`5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17`.
TRAIN suma 2.638.562 palabras Unicode: 1.600.747 de Wikipedia y 1.037.815 de
mensajes públicos OpenAssistant. Contar palabras no equivale a conocimientos
distintos ni acredita corrección de cada texto.

- Wikipedia: `wikimedia/wikipedia`, configuración `20231101.es`, revisión
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; se conservan títulos, identificadores
  y enlaces para atribución. Son datos nuevos para esta corrida de un snapshot
  de 2023, no noticias de 2026. Se filtraron encabezados, listas cortas, artículos
  sin suficiente prosa y texto con señales insuficientes de español.
- OASST1/OASST2: fuentes públicas fijadas y revisadas por sus etiquetas, con
  licencia declarada Apache-2.0. Se preservan las familias y particiones
  históricas; se filtraron fuentes con baja calidad, spam, etiquetas de datos
  personales o idioma incoherente. Los mensajes se deduplican por identidad.
- Se retiraron duplicados exactos y familias con texto normalizado compartido
  entre particiones. Este filtro no detecta toda paráfrasis o semejanza semántica.

El [manifiesto del corpus](foundation-15m-corpus-manifest.json) conserva revisiones,
hashes, filtros, cuentas y atribuciones. Mantiene por separado la declaración
CC-BY-SA 3.0/GFDL de la tarjeta del snapshot y la declaración actual CC-BY-SA
4.0/GFDL de Wikimedia; no certifica derechos artículo por artículo. No se usaron
mensajes personales del creador ni llamadas a un modelo para elaborar esta base.

El ajuste siguiente prepara diálogos humanos públicos y ficción sintética
identificada. Las composiciones sintéticas ejercitan hechos, nombres, roles,
correcciones y formulaciones; repetir estructuras no se presenta como ampliar
conocimiento general. Cada diálogo utiliza supervisión de la última respuesta:
una respuesta equivocada anterior puede ser contexto para corregirla, no un
objetivo positivo accidental. Sus cuentas y hashes finales se registrarán al
congelar y lanzar esa etapa; aquí no se da por entrenada.

La caché causal tiene 17.995 ventanas TRAIN y 2.189 VALIDATION; TEST contiene
cero ventanas. Cada pasada TRAIN incluye 5.780.408 posiciones objetivo reales,
con una media de 321,223 por ventana. No se supone que todas tengan 512 tokens.
La [contabilidad](stage-evidence.md) distingue exposición prevista de realizada;
el recuento exacto completado se obtiene del estado de checkpoint y sus índices.

## Ejecución y costes

El [diseño congelado](foundation-15m-stage-design.json) fija corpus, código,
configuración y límites. Se usa batch lógico ocho, microbatch cuatro, mmap,
cuatro hilos PyTorch y BF16 en CPU. La tasa base es 0,0006; warmup de 128 updates
y decaimiento coseno hasta el update 12.000, con mínimo de 10 % de esa tasa.
No se infiere un coste de GPU ni un servicio contratado.

El máximo persistente es **12.000 updates o 7.200 segundos acumulados**. Cada
invocación tiene hasta 3.600 segundos. El runner vigila **6.144 MiB de RSS** y
**2.048 MiB del directorio nativo** mediante parada cooperativa en un límite
completo del optimizador; no son barreras duras del sistema operativo. La
observación recoge 3.165.884 KiB de RSS del trabajador. No se atribuye al modelo
todo el espacio libre del filesystem como si fuera presupuesto de almacenamiento.

La NLL de validación se mide sobre las mismas 2.189 ventanas y 687.867 posiciones:

| Update | NLL de validación |
| ---: | ---: |
| 0 | 7,711048 |
| 1.024 | 4,555926 |
| 2.048 | 4,167366 |
| 3.072 | 3,773384 |

La caída refleja aprendizaje de predicción causal bajo esta métrica; no prueba
que conserve hechos o converse correctamente. El checkpoint seleccionado por
validación era step 3.072; el último reanudable observado era step 3.584.
No se copiaron pesos en escritura ni se presenta una exportación futura como
un artefacto ya publicado.

Por indicación del creador, esta fase prioriza entrenamiento y no ejecuta
baterías adicionales de software. La validación nativa del entrenador continúa.
Las 494 pruebas y 12 ejemplos del informe anterior corresponden a aquella
integración, no certifican los cambios nuevos de esta continuación.

## Bucle local y consulta de estado

```sh
python3 examples/continual_training.py status \
  --config private/learning-repair-20261013/configs/continual.json
```

La configuración de esta sesión se encuentra en una ruta privada excluida de
Git. No se crea al clonar. La [guía del supervisor](continual-loop.md) describe
cómo fijar otra configuración y añadir etapas con procedencia en el inbox.
El supervisor adoptó el trabajador existente; no duplicó su entrenamiento.
Está habilitado para reanudar `run_learning_stage.py --resume` al acabar una
invocación si queda presupuesto persistente y el checkpoint es coherente.
La próxima reanudación requiere observar sus eventos reales antes de declararla
comprobada. Adam, scheduler, orden de datos y RNG pertenecen a esa reanudación.
Una SFT nueva reutiliza pesos/tokenizador propios y crea otro optimizador.

El bucle puede esperar nuevas etapas sin un número máximo de ciclos mientras
el proceso y el entorno sigan activos. Cada corrida es finita y requiere corpus
nuevo identificado; sin entradas cambia a `waiting_for_data`. El colector de
datos automático está deshabilitado en esta observación. Ni un proceso dormido
ni repetir el mismo corpus indefinidamente se presentan como mejora progresiva.
No hay promesa de continuidad cuando termine el entorno.

Los presupuestos históricos 24+2 horas siguen cerrados. Esta continuación usa
la autorización posterior del creador; no reinicia tiempo acumulado ni elimina
los candidatos anteriores. El supervisor no promociona pesos por menor NLL y
mantiene `conversationCertification: awaiting_review`.

## Aceptación e integración pendientes

Tras SFT se fijará candidato y configuración antes de abrir una prueba nueva:
40 turnos en diez diálogos, mismo `ConversationEngine`, respuestas realmente
generadas como historia y condiciones equivalentes para C7 y candidato.
Se conservan el criterio de **36/40 respuestas completas**, **8/10 diálogos
totalmente correctos** y **cero fallos críticos o técnicos**. Los juicios de IA
del proyecto se identificarán como tales, conservando desacuerdos; no son
evaluación humana independiente ni comparación comercial equivalente.

GitHub confirma nueve ramas remotas en la observación y PR1–6 `MERGED`, sin
borradores. `main` y `work/conversation-ready-20261013` están en
`242aa3e57368352f72bea6704833f70be376b84c`; se conserva la
[reconciliación anterior](../conversation-integration-20261013/branch-reconciliation.json).
El trabajo nuevo está en `work/learning-repair-20261013` y aún no tiene commit
de entrega en esta lectura. El recibo posterior de integración debe identificar
commits remotos, corpus distribuible, tokenizador, pesos/checkpoints, hashes,
reanudación y conversaciones obtenidas. Este documento conserva el estado
observado durante entrenamiento, sin dar la misión por terminada.

## Avance añadido el 13 de octubre, 05:26 UTC

El cuerpo anterior conserva la observación de las 05:01 UTC. El
[recibo posterior](observation-20261013-052609-resume-sft-queued.json) añade este
avance sin reescribir [el recibo anterior](current-observation.json), cuyo
SHA-256 permanece
`fa490d8ed3593a083f5263deafd96ccb55fce5ebf6e605839674ed2935e6f6f5`.

La primera invocación terminó en step **4.360**, con **3.600,901 segundos**
nativos acumulados y causa `invocation_time_budget`. El
[recibo de reanudación](foundation-15m-first-resume.json) registra una salida
coherente, snapshot seleccionado en step 4.096 y `sourceAndInputsUnchanged: true`.
El supervisor lanzó otra invocación a las **05:11:32 UTC**, con `--resume`, PID
**11176** y nacimiento **1165743**. La reanudación usa el checkpoint nativo y
mantiene Adam, scheduler, orden, RNG y el límite de 7.200 segundos; no se
reinició el optimizador ni se abrió otra base aleatoria.

A las **05:26:09 UTC**, el trabajador seguía vivo y `running`, en step **5.460**,
época 2 y **4.468,758 segundos acumulados**. El supervisor **11159**, nacimiento
**1154726**, también estaba vivo. Las validaciones añadidas sobre las mismas
687.867 posiciones fueron:

| Update | NLL de validación |
| ---: | ---: |
| 4.096 | 3,466558 |
| 5.120 | 3,275227 |

La nueva lectura acredita avance del entrenamiento causal y la transición
real de reanudación; no acredita conversación competente ni una evaluación
de SFT. La validez de los PID sigue limitada al momento de la lectura.

### SFT fijado y encolado

`conversation-v1:conversation-15m` está en el inbox, con padre
`initial:foundation-15m`. El supervisor resolverá el snapshot seleccionado al
terminar esa base y comprobará su identidad. No hay un `status.json` de SFT en
esta observación y no se le atribuyen updates.

El [corpus preparado](../../examples/learning-repair-20261013/sft/sft.train-validation.jsonl)
ocupa **24.893.249 bytes**, con SHA-256
`4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510`.
Su [manifiesto](../../examples/learning-repair-20261013/sft/manifest.json) tiene
SHA-256 `a506f350df44fe81e34062d51c13a26fe41f874f48ea38363edc7dee9374d147`.

| Partición | Registros | Humanos públicos | Fracción humana |
| --- | ---: | ---: | ---: |
| TRAIN | 18.588 | 7.676 | 41,295 % |
| VALIDATION | 1.704 | 981 | 57,570 % |
| TEST en el archivo de entrenamiento | 0 | 0 | — |

El resto procede de ficción original y composiciones sintéticas con autoría
de agentes de IA del proyecto identificada; no son conversaciones humanas ni
respuestas generadas por Unidad. La mezcla se baraja uniformemente sin
reemplazo, pesos de fuente ni duplicación para alterar proporciones. La fracción
por registros no equivale a la fracción por tokens o a diversidad semántica.

La revisión editorial previa identificó **siete prefijos TRAIN y uno VALIDATION**
con inferencias o respuestas incorrectas. Se excluyeron como objetivos positivos,
conservando literalmente los mensajes equivocados como historia de los prefijos
correctivos posteriores. No se reescribió el diálogo fuente. Se retiraron además
**cinco diálogos exactos repetidos dentro de su misma partición**; no se movieron
familias y no se hallaron clases de diálogo o prompt completo compartidas entre
particiones bajo esos filtros. La revisión la hicieron asistentes de IA del
proyecto; no es certificación humana exhaustiva ni prueba de competencia.

La [configuración SFT](../../examples/learning-repair-20261013/configs/conversation-training.json)
fija `assistant_last_reply`, última respuesta y EOS como únicos objetivos,
reducción `sequence_mean`, ventanas `sliding_context`, mmap, todos los parámetros,
batch ocho/microbatch cuatro y cuatro hilos CPU BF16. La tasa base es 0,00015,
warmup 128 y coseno con mínimo 10 %. El límite es **8.192 updates o 7.200 segundos
acumulados**, con invocaciones de hasta una hora y los mismos techos cooperativos
de 6 GiB RSS/2 GiB nativos. Es otra etapa con optimizador nuevo a partir de pesos
propios; esa operación es distinta de la reanudación recién observada.

### Entradas y checkpoint preparados para entregar

El [recibo de distribución](distributed-inputs.json) conserva copias exactas de
corpus y configuraciones en `examples/learning-repair-20261013`. El
[corpus causal](../../examples/learning-repair-20261013/causal-base/causal-spanish.train-validation.jsonl)
mantiene el hash de la corrida activa. Los archivos preparados no incorporan
mensajes privados del creador y sus licencias se distinguen de la licencia del
código. Están locales; el recibo no afirma subida a GitHub.

La [plantilla pública](../../examples/learning-repair-20261013/continual-template.json)
contiene dos etapas finitas, base → SFT, sin adoptar el PID de esta sesión.
Antes de usarla hay que sustituir `/ABSOLUTE/PATH/UNIDAD` por la ruta del clon y
copiar la configuración a una ruta local. El supervisor crea sus estados en
`private/learning-repair-reproduction/`. Después de completar ambas etapas
espera contenido nuevo; `collector.enabled: false` no obtiene datos automáticamente.
El bloqueo exclusivo evita otro supervisor dentro del mismo workspace. Esta
reproducción no se ha ejecutado aquí y no se presenta como resultados equivalentes
ni garantiza continuidad fuera del entorno.

Se preparó una copia congelada del checkpoint nativo de **step 3.072** en
**nueve partes de hasta 32 MiB**, con 182.406.100 bytes nativos totales. El
[recibo de empaquetado](foundation-15m-step-3072-delivery.json) conserva hashes
antes/después, integridad de cada parte y carácter intermedio. Su
[guía](../../checkpoints/learning-repair-20261013/foundation-15m/README.md)
permite reconstruirlo en un directorio nuevo. En ese recibo
`restoreExecuted: false` y `gitHubUploadVerified: false`: preparar el paquete
no equivale a haberlo restaurado o subido, ni lo promociona como candidato
conversacional.

No se ejecutaron pruebas de software para este avance ni se abrió la suite
final reservada. El colector local permanecía deshabilitado; su activación
posterior necesitará otro registro real. El núcleo de entrenamiento sigue
congelado durante la etapa activa. El trabajo nuevo aún no está integrado en
`main`; un recibo posterior identificará sus commits, publicación y respuestas.

## Cierre de base y SFT activo, añadido el 13 de octubre, 06:23 UTC

Esta sección añade evidencia de ejecución posterior; no cambia las lecturas de
las 05:01 ni de las 05:26 UTC. El
[recibo nuevo](observation-20261013-062351-base-complete-sft-active.json) registra
el cierre de base, SFT activo, la ampliación en cola y el estado de GitHub.

### Resultado real de la etapa causal

`foundation-15m` cerró por `total_time_budget` en **8.914 actualizaciones**.
No alcanzó el máximo de 12.000. La
[contabilidad desde estado e índices](foundation-15m-final-target-exposure.json)
registra **71.297 ventanas** y **22.906.669 posiciones objetivo** procesadas:
tres épocas completas más 17.312 ventanas de la siguiente. Son posiciones de
exposición repetida, no vocablos o conocimientos distintos.

El umbral persistente era de 7.200 segundos. El contador final registra
**7.337,687718 segundos**, incluyendo validación y guardado de cierre. La
parada coopera con límites coherentes del optimizador y del entrenador; el
umbral no era un límite duro ni garantiza exactamente dos horas de pared.
Se conserva ese exceso observado en lugar de presentar el máximo configurado
como duración ejecutada. Código y entradas permanecieron sin cambios al cierre.

La NLL validada final fue **2,981873986** en step 8.914, sobre las mismas
687.867 posiciones usadas durante esta base. La bajada causal no se convierte
en mejora de conversación ni identifica por separado el efecto de datos,
capacidad, vocabulario o scheduler.

El [recibo de entrega final](foundation-15m-final-delivery.json) conserva el
artefacto de inferencia propio
`unidad-conversation-ee24651fb8cc3a01e1acbb57db5f40222a6b9b5fe150dbff7186b961945fa398`
en [models/learning-repair-20261013/foundation-15m/final](../../models/learning-repair-20261013/foundation-15m/final).
Sus cuatro archivos suman 60.784.468 bytes y conservan hashes antes, en la copia
y después. Los [dos checkpoints nativos finales](../../checkpoints/learning-repair-20261013/foundation-15m/final/README.md)
BEST y LAST tienen el mismo step, pero estados y nombres distintos; cada uno
se preservó completo en nueve partes de hasta 32 MiB. La copia se hizo con el
entrenador ya cerrado. Los recibos no atribuyen restauración ejecutada ni
publicación remota verificada a estos archivos finales locales.

### Fallo de transición y reparación observados

A las **06:13:53 UTC**, la transición del supervisor después de la base
terminada falló con
`TypeError: Supervisor.event() got multiple values for argument 'kind'`.
El argumento posicional de `event(self, kind)` chocaba con el campo `kind` de
la metadata del evento. El
[recibo de reparación](continual-stage-transition-repair.json), SHA-256
`48e50cdda5199f4892433c414c130de375d6e457fd7b4b250c5f03903fd630b7`,
conserva el error y la corrección: nombre posicional `event_name`, limpieza de
una razón operacional obsoleta y aceptación de alias `python`/`python3` sólo
cuando resuelven a la identidad del mismo ejecutable.

Se sustituyó **sólo el supervisor bloqueado** después de comprobar la base ya
completada; no se señaló un proceso de entrenamiento ni se reinició la base.
El núcleo ML congelado permaneció intacto. El nuevo supervisor **13378**,
nacimiento **1556040**, empezó a las 06:16:35 UTC y lanzó SFT real desde el
snapshot propio seleccionado. El fallo histórico se conserva; no se presenta
la transición como si hubiera funcionado desde su primera versión.

### Ajuste conversacional que ya actualiza pesos

En la lectura de **06:23:51 UTC**, `conversation-v1:conversation-15m` estaba
`running`, en **512 updates** y **402,289 segundos** nativos. Su PID **13379**,
nacimiento **1558057**, y el supervisor se comprobaron vivos. La etapa declara
`trainedFrom: own_checkpoint`, objetivo `assistant_last_reply`, todos los
**15.179.520 parámetros** entrenables y particiones `train_validation_only`.
El corpus conserva 18.588 TRAIN, 1.704 VALIDATION y cero TEST, con los hashes
y exclusiones documentados en el avance anterior. No se cambió el tokenizador.

La NLL inicial de esta etapa es una métrica SFT sobre otro objetivo y otros
datos; no se compara directamente con la NLL causal final. Todavía no se
generaron diálogos de desarrollo ni se abrió la reserva final. El hook de
generación DEV después de cerrar SFT está habilitado y ligado a fuentes/suite
por hashes, pero su ejecución aún no ocurrió. La certificación sigue en
`awaiting_review`; C7 no se sustituyó.

### Otra ampliación pública ya en cola

`wide-spanish-v1` está en el inbox con SHA-256 de manifiesto de ciclo
`7051a2fcc3bdac43012d9aa374e1cd7974a82135cbe8b52ce3798032e18b0099` y padre
`conversation-v1:conversation-15m`. Incluye una adaptación causal con nuevos
textos y una SFT posterior de retención desde el resultado causal. Cada etapa
conserva sus límites finitos; ninguna ha comenzado en esta lectura.

La [mezcla distribuible](../../examples/learning-repair-20261013/wide-spanish-v1/README.md)
contiene **4.084 TRAIN**, **621 VALIDATION**, **cero TEST**, **2.406 registros
nuevos** y **2.299 mensajes OpenAssistant de retención**. Ocupa 19.604.191 bytes,
con corpus SHA-256
`d3e3eb2ef6f8aa120a07c6ec334bd1e66c765cfd198696e700fef0778357ffd6` y
[manifiesto de shard](../../examples/learning-repair-20261013/wide-spanish-v1/wide-spanish.shard-001.manifest.json)
SHA-256 `4853a0867c5721eba2fc0dbe1be2313b98c4ceb27abc5689fa8ed2c07b26c488`.
Las fuentes nuevas suman **1.470.331 palabras entre TRAIN y VALIDATION**:

| Fuente nueva | Registros | Palabras | Género |
| --- | ---: | ---: | --- |
| Gutenberg | 177 | 482.039 | Literatura histórica, prosa y poesía |
| Wikisource | 85 | 241.877 | Ensayo histórico y gramática |
| Tatoeba | 1.432 | 710.328 | Frases comunitarias y traducciones |
| Wikcionario | 712 | 36.087 | Glosas, formas y ejemplos léxicos |

Las 277.063 palabras de las anclas OpenAssistant se declaran como retención,
no como texto nuevo. Tampoco se certifica que cada palabra de un documento
nuevo sea inédita: se conservan coincidencias del mismo fold registradas y
los filtros exactos no detectan toda paráfrasis. Se mantuvieron familias,
particiones, atribuciones, revisiones y condiciones por fuente; siete familias
nuevas se excluyeron mediante las guardas documentadas.

La fecha de obtención no sustituye a la fecha del texto: Tatoeba procede de un
export de octubre de 2026, pero las frases de esta porción inicial datan
principalmente de **2007–2012**. Literatura y ensayos conservan ortografía,
ciencia y opiniones históricas. Las [licencias](../../examples/learning-repair-20261013/LICENSE-DATA.md)
siguen por fuente, sin atribuir una única licencia a la mezcla. Estos subconjuntos
no representan todo el español ni acreditan conocimiento factual actualizado.

El colector local está ahora **habilitado**, con fuentes fijadas y límites, para
trabajar cuando falten etapas nuevas. Aún no se ejecutó en esta observación.
El bucle puede seguir mientras el entorno y supervisor estén vivos; cada
etapa es finita y puede detenerse o esperar datos/recursos. No se garantiza
mejora ilimitada ni continuidad cuando termine el entorno.

### Integración comprobada y parche local

[PR7](https://github.com/guadalupejuan001-code/UNIDAD/pull/7) está `MERGED` y sin
borrador. Su commit de trabajo es
`033148c6d0d09feefd6127f718b1dd4149cbad4c`; `main` remoto está en el merge
`32d08dacb070e349e8bbf3213133cb695d3e5f19`. GitHub Actions completó con éxito
los recibos de [trabajo](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38354851309)
y [main](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38355571276).
Esos checks corresponden a aquella versión; no se atribuyen al parche local
posterior de transición ni prueban conversación.

La reparación, los estados finales y la ampliación nueva aún esperan la siguiente
integración y publicación. No se hizo una batería de software, generación de
diálogo o evaluación final para este recibo. Se conservan los resultados
negativos anteriores y el criterio 36/40, 8/10, cero críticas/técnicas; finalizar
esta etapa causal no declara terminada la misión conversacional.

## SFT en ejecución y PR8 fusionado, añadido el 13 de octubre, 06:44 UTC

El [recibo posterior](observation-20261013-064409-sft-active-pr8-merged.json),
SHA-256 `fda0b816f766821f2ee1a1dab32e4bac67dd5ac941866cce9d24488b872f236d`,
añade estado real sin sobrescribir los recibos de las 05:01, 05:26 ni 06:23.
Se corrigió el hash transcrito en la referencia documental de reparación;
el archivo del recibo de reparación no se modificó.

`conversation-15m` tiene **2.200 actualizaciones** y **1.626,785730 segundos**
nativos en esta lectura. El entrenador 13379/nacimiento 1558057 y el
supervisor 13378/nacimiento 1556040 siguen vivos, con estado operacional
`training` y razón nativa vacía. El checkpoint del paso 2.048 tiene NLL SFT
validada **1,735812294**; la referencia inicial de la misma etapa fue
**2,738113565**. Se registra aprendizaje sobre el objetivo supervisado, sin
atribuir calidad de conversación ni comparar esa pérdida con la base causal.
No se generaron respuestas de esta etapa ni se abrió la reserva final.

[PR8](https://github.com/guadalupejuan001-code/UNIDAD/pull/8) está `MERGED`, sin
borrador, desde las **06:42:44 UTC**. Commit de trabajo:
`8a7dc3f43a5797fe81e0873ecbdd881c8e8a619f`; merge en `main`:
`34e5fd0c772305acdb18616c684c7c4cf7450cac`. Los pesos y BEST/LAST finales de la
base, la reparación del supervisor y los corpus nuevos ya se incorporaron a
`main`. La CI del commit terminó con éxito en
[PR](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38356784618)
y [push](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38356685472);
la [CI del merge](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38358257321)
seguía en ejecución en esta lectura. El recibo verifica hashes de los archivos
locales citados, pero no afirma una comparación independiente de sus bytes
remotos ni una restauración ejecutada de cada paquete.

La ampliación `wide-spanish-v1` continúa en cola desde los pesos propios del
SFT actual, sin haber entrenado todavía. Los hashes del corpus, manifiesto,
configuración activa y cola coinciden con las identidades fijadas. No se
modificó código ML, datos ni configuración activa para esta actualización.
Tampoco se ejecutaron pruebas de software o inferencia del modelo. El bucle
sigue dependiendo del entorno vivo y la aceptación permanece en revisión.
