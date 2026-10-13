# Estado comprobado de la continuación

Lectura del **13 de octubre de 2026, 09:14:40 UTC**. La conversación fiable y
la planificación aprendida siguen pendientes. Esta entrega conserva resultados
negativos, continúa con datos revisados e incorpora un controlador operativo
experimental; ninguna de esas tres cosas declara terminada la misión de Unidad.

## Pesos y respuestas

La base propia de 15.179.520 parámetros terminó con 8.914 actualizaciones y
22.906.669 posiciones objetivo procesadas. El primer SFT terminó con 8.192
actualizaciones lógicas, 65.524 ventanas consumidas y 5.286.178 posiciones
supervisadas incluido EOS. Su tiempo final fue 5.941,433658156999 segundos.
El límite configurado era `stopAfterStep=8192`: el runner registra
`absolute_step_boundary` y el estado nativo conserva `stopped`, `reason=signal`.
No se interpreta ese campo nativo como una nueva interrupción del trabajo.

BEST permanece en el paso 4.096, con NLL de validación
1,7110848086831454. Sus pesos, tokenizador y configuración son idénticos a la
entrega intermedia. LAST 8.192 se conserva por separado; no es la selección por
validación. Los [recibos finales](conversation-15m-final-delivery.md) y la
[contabilidad exacta](conversation-15m-final-target-exposure.json) fijan bytes,
posiciones y estados sin confundirlos con capacidad conversacional.

La [comparación con C7](comparison/README.md) contiene 97 JSON originales
copiados byte por byte, incluidos 80 turnos y sus historiales realmente
generados. Se usaron los mismos diez diálogos ficticios de DEV, temperatura
cero, semilla 42 y hasta 160 tokens por respuesta. Ambos revisores de IA del
proyecto obtienen:

| Modelo | Completas | Parciales | Fallidas | Diálogos completos | Críticas / técnicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| C7 propio conservado | 0/40 | 1 | 39 | 0/10 | 6 / 0 |
| SFT propio 15 M, BEST 4.096 | 0/40 | 1 | 39 | 0/10 | 12 / 6 |

**No hay mejora conversacional verificada.** C7 sigue como referencia propia,
también sin aceptación. No se resolvieron las discrepancias ocultándolas: en
SFT la primaria considera crítica `outing_activity_revision` y la cruzada
`cinema_attendee_reduction`, intercambiando esos dos juicios con igual total.
También se conservan las otras diferencias de dimensiones. Son revisiones de
IA del proyecto, no un panel humano independiente. La aceptación exige 36/40,
8/10 y cero críticas/técnicas en una evaluación nueva tras fijar candidato.
No se abrió la suite reservada para esta comparación.

LAST tiene además un artefacto propio para inferencia en
[`final-last-step008192`](../../models/learning-repair-20261013/conversation-15m/final-last-step008192/):
`unidad-conversation-f06b96e6c4c5149979b4846f9d03a4dd43dc8521dd13e3d5733bb164f90268f5`.
Su SHA de pesos es
`5fb3e10d8d4261c0bb704137d3e12540d2ed7e1d1979b8b40424bf3544f40c9c`.
El [recibo de exportación](conversation-15m-last-inference-export.json) identifica
ese artefacto. Su [generación y dos revisiones reales de DEV](comparison/development-sft-last8192-v1/README.md)
ya están publicadas: 0 completas, 3 parciales y 37 fallidas en ambos juicios;
0/10 diálogos completos. La primaria marca 11 críticas y la cruzada 12, ambas
con ocho técnicas. Sólo la cruzada considera crítica `tap_flow_correction`;
se conservan también las diferencias de relevancia en
`library_initial_deadline` y español en `quiet_scene_preference`. Las ocho
técnicas agotan tokens; cuatro de esas mismas salidas descartan contexto.
La [comparación estructurada](comparison/development-sft-last8192-v1/comparison-last-v1.json)
retiene IDs y juicios sin imponer consenso. No hay promoción de LAST.

## Siguiente entrenamiento

El [currículo natural + LATAM v4](../../examples/learning-repair-20261013/natural-latam-sft-v4/README.md)
conserva todas las filas humanas/curadas y reduce las procedurales a 1.598.
Tratamiento contiene 10.359 TRAIN/1.759 VALIDATION; control,
9.951 TRAIN/1.704 VALIDATION. La adición regional es ficción sintética atribuida
del proyecto: 463 registros, 408 TRAIN y 55 VALIDATION. Ambos brazos parten del
mismo BEST propio `072eedd2…`, con iguales configuración y límite de 4.096
actualizaciones. Se ejecuta tratamiento primero y control después. VALIDATION
difiere entre brazos; no se presenta su NLL global como comparación equivalente.

La [auditoría nativa de TRAIN](natural-latam-v4-train-audit/README.md) comprobó
10.359 ventanas, una por registro, 1.353.109 posiciones supervisadas incluidos
10.359 EOS, sin pérdida de prefijo, usuario actual o targets. La historia de
asistente anterior queda fuera de los objetivos. Las 408 filas regionales TRAIN
contienen 16.532 etiquetas supervisadas. No se leyó TEST para ese diagnóstico.

El primer intento de admisión se bloqueó por campos obligatorios de procedencia
y `testUsed` del manifiesto; un recibo anterior había omitido también
`approval.approved`. Se reparó la admisión con manifiestos compatibles sin
modificar los corpus, el entrenador, los límites cerrados ni los checkpoints.
Se reemplazó sólo el supervisor inactivo: PID 13378 por PID 19895,
`birth=2318340`; no se interrumpió un entrenamiento nativo activo.

La [activación del collector V2](collector-v2-enable-receipt.json) sustituyó
después el supervisor 19895 por 22557, `birth=2592201`, adoptando el mismo
entrenador 19896 vivo y sin señales, cambios de corpus/fuente/configuración
nativa ni reinicio de Adam/RNG/presupuestos. En esta lectura el supervisor registra `operationalState=training`,
`trainingActive=true`, `nativeWorkerAlive=true`. El proceso nativo del
tratamiento es PID 19896, `birth=2318347`, `status=running`, **step=3560**:
**3.560 actualizaciones de pesos comprobadas a las 09:11:51 UTC**, con
2.880,2897863769977 segundos
nativos acumulados. Su linaje fija el padre `072eedd2…` y SHA de pesos
`0a71ae3feb61b7dabac9f31ecc4fd312f97ead8a790ed13b8144617b60e3e4e6`.
El brazo control sigue pendiente. Ni iniciar un proceso ni encolar datos se
cuenta como aprendizaje ejecutado.

Esta lectura procede de `private/learning-repair-20261013/continual/state.json`,
SHA `2651b01ed77a1a2a461533b7fc8b1ff9258e332a2527af122191bb2e6daaf22b`,
y `runs/natural-latam-matched-v4/treatment/status.json`, SHA
`42703628deb262506f51f136c8155960ee841b7d6028db5dde86f639b27a3733`.
Son estados cambiantes del entorno; el README no garantiza que sigan vivos
después de esta fecha. Para leerlos de nuevo dentro del entorno:

```sh
python3 examples/continual_training.py status \
  --config examples/learning-repair-20261013/configs/continual-live-collector-v2.json
```

La [evaluación intermedia del tratamiento en 1.024](comparison/development-natural-latam-step1024-v1/README.md)
conserva 47 JSON byte idénticos: ambas revisiones de IA del proyecto dan cero
completas/parciales, cuarenta fallidas y cero diálogos completos. La primaria
marca una crítica en `cinema_initial_total`, frente a cero en la cruzada:
«Tus entradas en dos entradas.» puede leerse como cantidad errónea o como
fragmento sin afirmación inequívoca. Se conservan ambos juicios. Hay once
técnicas: siete token_budget y cinco descartes, con una salida común. Treinta
+y tres terminan por EOS, sin excepciones ni respuestas vacías. No hay mejora
conversacional verificada ni aceptación reservada. Hasta esta lectura, la mejor
NLL propia del tratamiento es 1,7603046832825393 en 1.024; no se compara con
la VALIDATION de otro corpus. El [estado y cola del bucle](loop-runtime.md)
separan esta corrida del control pendiente y de las etapas futuras.

## Datos públicos nuevos

El [inventario directo regional](../../examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/README.md)
reúne 22.555 pertenencias a categorías de 19 países/territorios, correspondientes
a 11.241 entradas únicas. La muestra adquirida de definiciones tiene 120
entradas: 106 TRAIN/VALIDATION con 229 acepciones y 14 TEST privadas excluidas.
El catálogo causal original tiene 10.130 filas TRAIN/VALIDATION de metadatos.
La pertenencia de un título a una categoría no identifica todos sus significados
ni prueba que cada acepción sea exclusiva de un país.

La [mezcla causal amplia + LATAM v2](../../examples/learning-repair-20261013/wide-latam-v2/README.md)
retiene 10.129 filas del catálogo tras excluir una familia conflictiva. Incluye
también prosa y definiciones regionales, con fuente, licencias y particiones.
Esa mezcla y el [lote posterior de Tatoeba](../../examples/learning-repair-20261013/new-sources/tatoeba-spanish-next-v1/README.md)
están preparados pero **no entrenados**. La revisión causal v3 ya corrigió
glosas dudosas y [se encoló](wide-latam-v3-enqueue.json) después de acciones,
con dos shards causales y SFT de retención; [Tatoeba posterior también está
encolado](tatoeba-next-v1-enqueue.json). Estas etapas aún no se entrenaron. El SFT regional v4 ya excluyó los
targets dudosos detectados y conserva v2/v3 como evidencia. La cobertura sigue
siendo una muestra; no se afirma haber adquirido ni aprendido todo el español
latinoamericano. Las anclas repetidas son retención, no conocimiento nuevo.

## Autonomía operativa experimental

La API del cerebro incorpora `autonomy.start/status/step/run/cancel/approve/resume`.
El modelo conversacional propio propone acciones estructuradas y el controlador
comprueba herramientas, argumentos, permisos, límites y resultados reales. Guarda
progreso e intentos para recuperación; una aprobación sensible usa una credencial
de operador separada. Los directorios y herramientas se fijan al conectar el
producto, y el contenido de una petición no puede ampliarlos.

La [demostración de software](autonomy-demonstration/fixture-v1/evidence.json)
ya ejecutó la API autenticada sobre archivos reales:
leyó/inspeccionó archivos ficticios, guardó un informe verificable, canceló otro
intento y conservó los originales por hash. Sus acciones fueron una
**fixture declarada del controlador**, no un plan aprendido. Su evidencia tiene
SHA `2f5dc77ac42359b4d3cd043bc8e632566f284d44bf3e635914efff869acd9ea7`.
Los [dos intentos reales del mismo BEST propio 4.096](autonomy-demonstration/own-model-v3/evidence.json)
se conservan completos, SHA
`865eab21c96642dda24bb2014c943d6a05769c30fcac228c05cf3560b1690b3a`.
La tarea con tres herramientas necesitaba 732 tokens para un contexto de 512:
el controlador registra 220 tokens que se perderían y bloquea antes de generar,
`blocked_context`, un intento fallido, sin informe del modelo. La tarea mínima
necesitaba 336 tokens y no descartó contexto; el propio modelo produjo exactamente:

```text
"Lo siento, no puedo ayudarte. Si tienes alguna otra pregunta, no dudes en hacerla. Estoy aquí para ayudarte.
```

La comilla inicial queda sin cierre; tampoco es un objeto de acción. La salida
terminó por EOS, con 35 tokens nuevos, pero es JSON inválido. El controlador
registra `blocked_model`, un intento fallido y cero herramientas ejecutadas por
el modelo. El informe `modelo.json` no existe. Ninguna fixture sustituye la
propuesta fallida. `planningAccepted=false` y las capacidades de planificación
natural/trabajo del modelo siguen sin aceptar. Los registros de experiencia
no actualizan pesos ni activan reentrenamiento.

La verificación de cierre pasó 143 pruebas en 6,393 segundos, con una CPU y
límite de 180 segundos. Incluye autonomía/controlador/API/archivos, contratos
del cerebro anterior, bucle, collector V2, nueve pruebas de extensión y seis
de datos de acciones. Es evidencia de software; no cambia los fallos del modelo.

La [extensión realmente exportada](context2048-extension-delivery.json) es
`unidad-conversation-ba70fff6928dcaf08087a4c966c1ea6496555bc04683933789908b4d8f49501c`,
con contexto 2.048 y 15.769.344 parámetros, desde el tratamiento propio 1.024.
Hereda todos los tensores y las primeras 512 posiciones byte por byte, conserva
el tokenizador `848a8d76…`, y añade 589.824 posiciones parametrizadas sin entrenar.
La [prueba real de las tres herramientas con ese artefacto](autonomy-demonstration/context2048-baseline-v1/evidence.json),
SHA `f483eed5ce69cc09f4b63e8384f80e322e0a524b785c38a7de1c313ac0880d43`,
ya conserva los 732 tokens de prompt, sin descarte. Genera 52 tokens y termina
por EOS, pero el texto real es JSON inválido:

```text
"}, "fil":fil=do.or:fil=do.or:fil=do.or:fil"}:fun("]:fil=do.or:fun("
```

Termina `blocked_model`, sin informe del modelo ni fallback, con originales
sin cambios. Reparar el contexto inicial no demuestra que haya aprendido acciones.

El [corpus de acciones](../../examples/learning-repair-20261013/autonomy-action-sft-v1/README.md)
registra 72 tareas ficticias, 216 acciones reales de archivos y 192 TRAIN/24
VALIDATION separados por tarea/familia. Las etiquetas JSON son fixtures declaradas
CC0 de IA del proyecto, no propuestas aprendidas ni pesos de otra IA. El contexto
2.048 conserva la supervisión de hasta 1.268 tokens. Su [SFT encolado](autonomy-action-v1-enqueue.json)
parte del artefacto ampliado, después de tratamiento y control, con límite de
256 actualizaciones/600 segundos, lote 2 y microbatch 1. **Aún no está entrenado**.
La retención natural + acciones conserva todas las filas anteriores y contiene
10.551 TRAIN/1.783 VALIDATION, sin contar esa repetición como conocimiento nuevo.
La [cola y los límites](loop-runtime.md) también fijan las etapas causales v3,
retención, Tatoeba y collector habilitado, sin prometer continuidad fuera del
entorno o mejora automática.

## Git y publicación

El trabajo local está en `work/learning-repair-20261013`, HEAD
`91e7f3a84bccc968df97be717a1e45e17519fe67`. A las 08:24:47 UTC se comprobó
`origin/main=3930c1ef5c25ff5953a5b57793231e7f78c6eac7`, con
[PR9 fusionado](https://github.com/guadalupejuan001-code/UNIDAD/pull/9).
Los nuevos cambios de controlador, cierre y evaluación de esta continuación
están **sin commit en esta lectura**; aún no se presentan como integrados en
`main`. Los recibos históricos conservan la fecha y el estado de su publicación.
Clonar el repositorio no inicia entrenamientos ni mantiene el supervisor vivo.
