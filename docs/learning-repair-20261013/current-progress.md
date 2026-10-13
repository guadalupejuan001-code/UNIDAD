# Estado comprobado de la continuación

Lectura del **13 de octubre de 2026, 11:28:38 UTC**. La conversación fiable y
la planificación aprendida siguen pendientes. Esta entrega conserva resultados
negativos, continúa con datos revisados e incorpora un controlador operativo
experimental; ninguna de esas tres cosas declara terminada la misión de Unidad.

## Pesos y respuestas

La [evaluación posterior de retención 1.024](comparison/development-wide-latam-retention1024-v1/README.md)
terminó sobre el propio `3ccd6deb…`: **1/40 completa y 0/10 diálogos** para
ambos jueces. Primaria: cuatro parciales/35 fallidas; cruzada: una parcial/38
fallidas. Ambas: seis críticas/dos técnicas, con los mismos IDs. La completa es
`library_initial_deadline`; tres diferencias partial/failed y ocho diferencias
totales de dimensiones/grado se conservan. Las 52 copias exactas incluyen
respuestas por turno, juicios, assessments, freeze y metadatos de artefacto;
el proof verifica historiales reales sin descartes ni excepciones.

El [endpoint antes de esas etapas, ACTION 256](comparison/development-action2048-best256-v1/README.md)
da 0/40 completas,0/10 diálogos, cinco críticas/seis técnicas en ambos juicios;
primaria una parcial/39 fallidas, cruzada tres parciales/37 fallidas.
Se preservan cinco discrepancias de dimensiones/grado y 52 copias exactas.
Ambos usan la misma rúbrica, contexto 2.048, 160 tokens, temperatura 0/semilla 42
e historial realmente generado. Cambian múltiples etapas/pesos/corpus; no
se atribuye el cambio a una sola variable ni a NLL de objetivos distintos.
Son IA del proyecto y DEV expuesto, sin aceptación. Los resultados anteriores
0/40 y el probe regional 0/12 conservan sus archivos originales.

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
nativa ni reinicio de Adam/RNG/presupuestos. En esta lectura, tratamiento y control ya están cerrados.
Acciones2048 también llegó a **256 updates**, `stopped`, PID 25371,
`birth=3029565`, tiempo 187,0496777519984 segundos, todos los 15.769.344
parámetros entrenables. BEST 256 tiene NLL de acciones 0,03811874364813169.
Su [entrega final](autonomy-action2048-final-delivery.md) conserva 36.385
posiciones supervisadas, incluidas 512 EOS, en 512 ventanas.

La **base causal amplia español + LATAM v3 terminó ambas etapas**. LANG1,
PID 25651/`birth=3078774`, cerró 2.048 updates en 1.931,1925520530021 segundos,
BEST 2.048 NLL 3,2358626570786604. Parte del ACTION 256 propio `e202eafc…`.
Su [entrega final](wide-latam-v3-language-1-final-delivery.md) conserva
1.445.480 posiciones supervisadas, 3.921 EOS y 4.096 ventanas consumidas;
quedaron 6.110 de las 10.206 ventanas TRAIN sin consumir. El umbral de 1.800 s
era cooperativo: se excedió en 131,19 s; motivo nativo `signal`, runner
`absolute_step_boundary`. BEST y LAST tienen pesos iguales, estados distintos.
El [artefacto de inferencia completo](wide-latam-v3-language-1-inference-export.json)
es `unidad-conversation-a8c25b012d4cbf1da6b818983dc424ec219b93b7793115bb5d6474d8ef3087f8`,
SHA de pesos `b8856eed20322b4c0011b274fe582a3b9ff6b1ab57d874cb300420a617b5dc69`.

LANG2, PID 26737/`birth=3273905`, cerró 2.048 updates en 486,66538323400164 s,
BEST 2.048 NLL causal 2,143346329509529. Retención natural + acciones,
PID 27053/`birth=3323970`, cerró 1.024 en 638,6049642170037 s, BEST 1.024
NLL SFT 1,7224484366660942. Ambos conservan `stopped`, motivo nativo `signal`.
Los corpus/objetivos son distintos: esas NLL no son mejora conversacional.

El snapshot procede de `continual/state.json` y los cuatro `runs/*/status.json`.
El supervisor 22557 registra `generating_development`, `trainingActive=false`,
`nativeWorkerAlive=false` y `generationChildAlive=true`, PID 27186/`birth=3420096`.
Genera DEV del tratamiento V2 ya cerrado; los recibos anteriores conservan
sus hashes y fechas, sin reescribirlos.
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
separan los brazos terminados, acciones256 y las etapas causales futuras.

El [tratamiento terminó](natural-latam-v4-treatment-final-delivery.md) con
4.096 actualizaciones y 3.337,367299398 segundos, al alcanzar su límite
configurado (`absolute_step_boundary`; estado nativo `signal`). Procesó
32.765 ventanas, 4.277.235 posiciones supervisadas, incluidas 52.188 regionales
en 1.295 exposiciones regionales. El
[recibo final](natural-latam-v4-treatment-final-delivery.json), SHA
`89079a6a314024ff6208837d49d39be12f30e63d29cbf2d45982b9b813a71e1e`,
conserva fuentes, presupuestos y estados históricos. BEST sigue en 1.024,
NLL 1,7603046832825393. Sus pesos/configuración/tokenizador son idénticos a
`intermediate-step001024`; los metadatos del manifiesto de selección final son
distintos y se conservan aparte. La carpeta de referencia final no se carga
como si fuera un artefacto de inferencia completo.

El paquete nativo LAST 4.096 contiene 182.360.596 bytes en nueve partes. El
[artefacto LAST para inferencia](../../models/learning-repair-20261013/natural-latam-v4/final-treatment-last-step004096/)
es `unidad-conversation-d6d860222baf5e8952eade42fcda3a0bd52dc77000c0c254ca895a91fdab7160`,
SHA de pesos `5888240335df305a5f80aa438200b524af741278d8e8c7af234d8a60f86f7f6e`.
Ya generó cuarenta turnos DEV reales, con cero excepciones. La
[copia de generación](comparison/development-natural-latam-last4096-v1/README.md)
conserva historial y metadatos; outputs tiene SHA
`8011eefafab0b823fe49fe689607f8225eb8c36a7a9b486c37bbe8b28ba114ac`.
Las revisiones primaria y cruzada terminaron: ambas dan **0 completas, 1 parcial,
39 fallidas, 0/10 diálogos, 14 críticas y una técnica**. La única parcial es
`library_initial_deadline` y el técnico `library_day_correction`, token_budget
sin descarte de contexto. Se comprobaron los IDs críticos, que son los mismos
catorce; el juicio de español difiere en `cinema_attendee_reduction` y
`outing_one_sentence_final`. La primaria incluye auditoría auxiliar de siete
casos mediante IA del proyecto; ninguna revisión es humana independiente.
Los 45 JSON de generación/metadatos y cuatro juicios/assessments se prepararon
localmente para una publicación posterior a PR10. LAST no es BEST ni un modelo
aceptado. Esta intervención tampoco verificó mejora conversacional. Procesar targets y entregar paquetes no acredita
conversación competente.

El [control terminó en 4.096](natural-latam-v4-control-final-delivery.md), tiempo
3.321,1108099389967 segundos y razón nativa `signal`. Su exposición exacta es
4.402.874 etiquetas supervisadas, incluidas 32.765 EOS, en 32.765 ventanas.
BEST del control es **0**, idéntico al padre `072eedd2…`: heredó pesos ya
entrenados, pero tiene cero posiciones y updates nuevos de control. LAST propio
es `unidad-conversation-a2cae3b623142153487760d2f09e1a9c2fc851a5a1a09f1cad83b9ffaa6c2d7d`,
con [artefacto completo](../../models/learning-repair-20261013/natural-latam-v4/control-final-last-step004096/).
La [generación conversacional del control](comparison/development-natural-control-last4096-v1/README.md)
ya terminó cuarenta turnos sin excepciones, SHA de outputs
`585e4166e4fdb6c5af3a5cd25c636c700327ea9b073cee05c195bc186518fa52`.
La copia local conserva 45 JSON de generación/metadatos y cuatro de juicios.
Ambas revisiones de IA dan **0 completas, 1 parcial, 39 fallidas, 0/10 diálogos,
2 críticas y 7 técnicas**. Los críticos `outing_day_revision` y
`outing_one_sentence_final` y las cuatro claves booleanas coinciden por turno;
se comprobó directamente. Se mantienen notas/métodos completos y aceptación falsa.

El [probe regional de cuatro brazos](comparison/latam-knowledge-development-probe-v1/README.md)
conserva sesenta respuestas: padre, tratamiento BEST 1.024, tratamiento LAST
4.096 y control LAST 4.096. Ambas revisiones obtienen **0/12 significados TRAIN
y 0/6 familias en cada modelo**. Sólo el padre expresa incertidumbre apropiada
en 1/3 casos VALIDATION; los otros, 0/3. VALIDATION no cuenta como conocimiento
aprendido. Agotamientos de 80 tokens: 8/7/3/8; sin excepciones, vacíos o descartes.
Se publican 92 archivos byte idénticos, con freeze, soporte, prerregistro y la
vinculación previa del endpoint de control. La primaria es del mismo autor del
probe, con ese vínculo declarado; la cruzada es de otro agente de IA. No son
juicios humanos independientes. En `val07u` del control difiere la clasificación
semántica; se conservan ambos juicios y sus políticas técnicas. El probe pequeño
no acredita cobertura total del español ni mejora léxica, y no abrió TEST ni
una suite reservada.

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
encolado](tatoeba-next-v1-enqueue.json). Ambos shards v3 terminaron 2.048
updates cada uno y retención, 1.024; Tatoeba sigue en cola. El SFT regional v4 ya excluyó los
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

La verificación de cierre pasó 145 pruebas en 6,893 segundos, con una CPU y
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
256 actualizaciones/600 segundos, lote 2 y microbatch 1. **La etapa ya realizó
256 actualizaciones reales**, con todos los parámetros entrenables. El artefacto
extendido inicial conserva su estado histórico sin entrenamiento de posiciones
nuevas; el candidato de acciones es una etapa distinta, con pesos propios
entrenados. Su evaluación real se incorpora por separado; aún no se acepta
planificación, trabajo fiable o conversación de ese candidato. La [demostración real de acciones256](autonomy-demonstration/action256-own-v1/README.md)
conserva las tres herramientas genéricas en 732 tokens. La primera respuesta
propia fue JSON válido, literalmente:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"k.txt"},"reason":"Medir"}
```

`k.txt` no pertenece a `dos.txt`/`uno.txt` autorizados. El controlador lo bloqueó
antes del handler, `blocked_policy`, un intento/fallo, sin informe del modelo.
No se corrigió el nombre ni se insertó una respuesta prefabricada. Se observa
mejora parcial de formato en esa muestra frente al baseline 2.048 mal formado,
pero el trabajo no se completó. La tarea mínima dio JSON inválido y
`blocked_model`. La evidencia exacta tiene SHA
`ef2fa9da8ece2b9550cf59521a8369ae46ea21cedae2fff82eac19b8d7270789`;
se publicaron sólo cuatro archivos seguros, sin SQLite/memoria, preservando
los originales y el informe de fixture, no del modelo.

El [reintento opcional del host](operational-autonomy.md) ya permite una nueva
propuesta después de argumentos rechazados de una lectura, antes de iniciar
cualquier efecto, con `max_argument_repairs=1`; el valor por defecto es cero.
Conserva fallo, pasos y presupuesto; no cambia argumentos ni permisos. Su
verificación específica pasó 84 pruebas en 1,653 segundos, incluidas seis
regresiones nuevas. La [prueba real con esa opción](autonomy-demonstration/action256-own-read-repair-v1/README.md)
ya terminó: dos pasos, dos fallos y una reparación. Conservó `k.txt` sin editar,
y la segunda propuesta fue JSON mal formado; `blocked_model`, sin efectos,
informe o fallback. Los originales permanecen iguales; los pesos no cambiaron.

La retención natural + acciones conserva todas las filas anteriores y contiene
10.551 TRAIN/1.783 VALIDATION, sin contar esa repetición como conocimiento nuevo.
La [cola y los límites](loop-runtime.md) también fijan las etapas causales v3,
retención, Tatoeba y collector habilitado, sin prometer continuidad fuera del
entorno o mejora automática.

El [ciclo V2 de copia de argumentos](autonomy-action-copy-v2-enqueue.json)
ya entrenó tratamiento: PID 27145/`birth=3403056`, `stopped`, 256 updates,
157,4759767220021 s, motivo nativo `signal`; BEST 256 NLL 0,32269909729560214.
El modelo propio `e70ea28e…` está generando DEV. El control V1 aún no tiene
estado nativo en este snapshot; la comparación semántica está pendiente.
V2 y V1 parten de los mismos pesos `e202eafc…`, 256 updates/600 segundos por brazo,
192 TRAIN/24 VALIDATION con 72 tareas y 216 acciones reales. La preparación
aumenta la fracción de targets de nombres de archivo, 20,69 % frente a 13,34 %;
no representa mejora aprendida. Los casos de la demostración no son corpus.
Se ejecuta tras los dos shards v3 y retención ya cerrados, conservando el padre
original de Tatoeba. Las [instrucciones](operational-autonomy.md) distinguen
JSON válido, argumentos autorizados y finalización verificada. No se acepta
la capacidad por loss ni se inventan estados antes de que una etapa comience.

El [intento propio V2](autonomy-demonstration/action-copy-v2-treatment-own-v1/evidence.json),
SHA `d4a6c742f05a517240712476dec1e277b0569e8fe3ebd5738199fafb3f143d52`,
del 13 de octubre a las 11:27:59 UTC, repitió la misma tarea expuesta excluida
de TRAIN. Primero propuso literalmente `inspect_file` sobre `dos.txt`,
con JSON válido; el handler se ejecutó y la lectura se verificó de forma
independiente. Frente al `k.txt` bloqueado de E202, se observa copia autorizada
en este caso. Luego generó JSON inválido/repeticiones hasta 128 tokens, prompt
933 sin descartes: `blocked_model`, dos pasos/un fallo, sin informe del modelo.
Los originales permanecen iguales. La tarea mínima 336 tokens también dio
JSON mal formado hasta 128. No es generalización ni planificación aceptada,
y el control del contraste sigue pendiente. La fixture continúa declarada aparte.

El adaptador de acciones también corrige un bloqueo del encoder por marcadores
de rol literales dentro del JSON observado. Escapar `<`/`>` como `\u003c`/`\u003e`
conserva los valores exactos al decodificar; no omite datos ni cambia pesos.
La verificación pertinente aprobó 87 pruebas, incluidas tres regresiones de
serialización. Es software comprobado, no una nueva ejecución propia exitosa.
El [recibo posterior de lectura real](autonomy-lossless-observations-real-read.json)
de las 11:32:33 UTC conserva el archivo original de 710 bytes y las
observaciones exactas al decodificar. Usa motor fixture declarado, sin cargar
pesos propios ni entrenar; separa el IO verificado del aprendizaje.

## Git y publicación

Evidencia posterior al registro de las 11:28:38 UTC: el [contraste operativo](autonomy-action-copy-v2-real-matched-development.json)
de las 11:37:03 confirma ambos brazos cerrados 256, tratamiento 157,475977 s y
control 131,484277 s. Con primer prompt/archivos/herramientas/decodificación
idénticos, padre E202 pide `k.txt`, control 734c `v.txt`, ambos bloqueados antes
del handler; tratamiento E70 inspecciona `dos.txt` realmente verificado y luego
falla por JSON inválido. Ningún informe propio, originales iguales; mejora
parcial en un caso expuesto, sin generalización ni planificación aceptadas.
En la lectura posterior del supervisor del **13 de octubre,11:41:18 UTC**,
Tatoeba causal está `running`, PID 28222, paso 0,3,6537 s. Es inicialización,
sin nuevas actualizaciones del optimizador confirmadas en ese corte. Los
registros anteriores conservan su timestamp y estado; no se reescriben recibos.

[PR10 está fusionado y nunca fue borrador](https://github.com/guadalupejuan001-code/UNIDAD/pull/10).
Su head final fue `88efe444bdb4bceee07e298790abb12b76322fc6` y el commit de
integración en main es `62cf4d43780204f8d09dce6715cefcb6110aa310`. Los cuatro checks de las ejecuciones
`38379096070` y `38379113739` terminaron correctamente.
El [recibo de publicación](pr10-publication-verification.json) identifica
objetos Git remotos y SHA completos de código, configuración, corpus y pesos
propios. El [log final de GitHub](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38379113739)
registra 603 pruebas de conversación aprobadas en 75,599 segundos y doce
pruebas de ejemplos aprobadas en 7,312. Python mínimo informa 521 pruebas en
8,807 segundos, con 166 omisiones opcionales y resultado OK. Integrar esos
bytes y aprobar software no acepta conversación ni autonomía aprendida.

El [recibo de correcciones](pr10-ci-corrections.json) conserva el workflow
inicial fallido `38375885022` y sus causas. El control de archivos fija
dispositivo, inode, tamaño, mtime, ctime y modo para rechazar reemplazos incluso
si se reutiliza un inode; la aserción original de seguridad permanece. La prueba
del journal conserva seis payloads exactos de 200.000 caracteres y más de
1 MiB persistido; sólo su cliente usa 20 segundos dentro de la tarea de 30,
en lugar del timeout de tres incompatible con ese caso. Los fallos anteriores
siguen registrados como fallos. Fuente, datos y estado nativos de entrenamiento
permanecen congelados y los hashes históricos de acciones no se reescribieron.

[PR11 está abierto sin borrador](https://github.com/guadalupejuan001-code/UNIDAD/pull/11),
head publicado `68b6af16d1056c2ef02d013144cf3501b81c9ff3`, con los cuatro
checks de `38387474574`/`38387694065` en SUCCESS, comprobados en GitHub.
Control, acciones y probe regional están en esa entrega; main conserva PR10.
La segunda entrega con LANG1, serialización y este snapshot está local,
pendiente de publicar y de su CI; no se atribuye al head anterior.
Los recibos históricos conservan su
fecha y estado. Clonar el repositorio no inicia entrenamientos ni mantiene el
supervisor vivo. El collector se activa tras las etapas en cola cuando entorno
y recursos lo permiten; ninguna etapa promueve pesos automáticamente.
