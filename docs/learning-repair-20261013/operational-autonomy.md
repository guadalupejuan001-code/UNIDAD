# Autonomía operativa experimental

El flujo está integrado en `Brain` y en `/v1/requests`: objetivo, propuesta JSON
de los mismos pesos propios, validación del controlador, herramienta autorizada,
observación real y verificación independiente. La API conserva las operaciones
de conversación, herramientas explícitas y workflows anteriores.

`AutonomyController` conserva tareas y acciones en SQLite por BodyID. No comparte
esas tareas al compartir una memoria de conversación. Comprueba identidad de
modelo y políticas al reanudar, restaura lecturas verificadas contra archivos
actuales y evita repetir escrituras de resultado incierto. La cancelación impide
nuevas acciones. Un callback externo ya iniciado necesita su propio timeout;
cancelar no revierte efectos realizados.

El host registra herramientas con validador y efecto explícito. Destrucción,
publicación y gasto requieren aprobación para la acción y argumentos concretos,
con nonce y credencial de operador separada. Esas credenciales no se entregan al
modelo ni se guardan en su progreso. Las peticiones no eligen raíces, permisos,
herramientas, modelo ni presupuestos. Los límites incluyen tiempo, pasos,
fallos, tamaño de propuestas, resultados, estado y archivos.

## Evidencia ejecutada

La [demostración por API](autonomy-demonstration/own-model-v3/evidence.json)
ejecutó tres herramientas reales: medir un archivo, leer otro y guardar un
informe nuevo. El informe se construye desde observaciones medidas, no desde
texto inventado por el proponente. Los SHA de los originales permanecieron
iguales. Otra tarea cancelada después de una lectura no creó su informe.
**Esas propuestas están identificadas como `test_fixture`: demuestran el
controlador, no planificación aprendida.**

Con los pesos propios `072eedd…`, el primer objetivo con tres herramientas
necesita 732 tokens en una ventana de 512. Termina `blocked_context` antes de
generar. El objetivo reducido de inspeccionar `uno.txt` usa 336 tokens y sí
ejecuta inferencia propia. La respuesta real fue:

> "Lo siento, no puedo ayudarte. Si tienes alguna otra pregunta, no dudes en hacerla. Estoy aquí para ayudarte.

La comilla inicial sin cierre pertenece a la respuesta. No es JSON de acción:
la tarea termina `blocked_model`, sin ejecutar herramientas. No se sustituyó por
un plan prefabricado. `naturalLanguagePlanner` y `modelProposalValidated`
permanecen falsos. La extensión de posiciones y la supervisión de acciones
constituyen intervenciones posteriores; prepararlas no acredita aprendizaje.

## Resultado real después de supervisar acciones

El [cierre del SFT de acciones](autonomy-action2048-final-delivery.md) conserva
256 updates, 187,0496777519984 segundos, 512 ventanas/EOS y 36.385 posiciones
supervisadas. Todos los 15.769.344 parámetros eran entrenables. El artefacto
propio [BEST 256](../../models/learning-repair-20261013/autonomy/action2048-final-best256/)
es `e202eafc…`; la NLL de acciones 0,03811874364813169 no acredita trabajo fiable.

La [demostración posterior con esos mismos pesos propios](autonomy-demonstration/action256-own-v1/README.md)
conserva las tres herramientas genéricas y el prompt completo de 732 tokens.
La primera respuesta real, sin corregir `k.txt`, fue:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"k.txt"},"reason":"Medir"}
```

Es JSON válido y termina por EOS tras 63 tokens. Los archivos autorizados eran
`dos.txt` y `uno.txt`: `k.txt` queda rechazado **antes del handler**,
`blocked_policy`, un intento/fallo, sin informe del modelo. Frente al
[baseline 2.048 anterior](autonomy-demonstration/context2048-baseline-v1/evidence.json),
que produjo JSON inválido para el mismo marco, esta muestra observa mejora
parcial de formato. No demuestra selección correcta de archivos ni trabajo
completado. La tarea mínima aún produce JSON mal formado y `blocked_model`.

La [evidencia original exacta](autonomy-demonstration/action256-own-v1/evidence.json)
tiene SHA `ef2fa9da8ece2b9550cf59521a8369ae46ea21cedae2fff82eac19b8d7270789`.
Se conservan peticiones, respuestas y huellas reales, originales sin cambios y
el informe únicamente de la fixture declarada. No se publicó SQLite ni memoria,
no hubo fallback y `planningAccepted=false`. La propuesta propia no se sustituyó
por una secuencia de herramientas del controlador.

## Nueva propuesta tras un argumento de lectura rechazado

El host puede habilitar `--autonomy-max-argument-repairs 1` en CLI, o
`AutonomyLimits(max_argument_repairs=1)` en Python. El valor por defecto es **0**.
Permite al modelo formular una propuesta nueva después de que se rechacen los
argumentos de una herramienta **de lectura**, antes de iniciar cualquier efecto.
No reescribe la respuesta previa ni cambia `k.txt` por otro nombre. El fallo
permanece registrado y consume los límites de pasos/fallos; el permiso, raíces,
archivos autorizados y presupuesto se comprueban otra vez. No se aplica a
escrituras, acciones sensibles, efectos iniciados ni aprobaciones pendientes.

La verificación específica aprobó 84 pruebas en 1,653 segundos, incluidas seis
regresiones de esta opción. La [prueba posterior del modelo con un reintento](autonomy-demonstration/action256-own-read-repair-v1/README.md)
ya se ejecutó: conservó la propuesta de `k.txt`, registró el rechazo y recibió
otra propuesta mal formada. Terminó `blocked_model`, dos pasos/dos fallos/una
reparación, sin efectos ni informe, con originales sin cambios. Su evidencia
exacta tiene SHA `72886bdb689cde178b6134ed1a799f8dfd9d3620fdf7bb373cfa6b63d2c5fe91`.
El reintento de software funciona; este modelo no completó la tarea. Los
intentos anteriores conservan el límite original de cero reparaciones.

## Datos JSON que contienen marcadores de rol

El adaptador tenía un defecto antes de generar: un archivo o dato de tarea
con marcadores literales como `<system>` bloqueaba el encoder de conversación.
Ahora, si el JSON serializado contiene `<`, escapa todos los `<` como `\u003c`
y los `>` como `\u003e`. `json.loads` recupera exactamente los valores originales,
incluidos escapes literales; los mensajes sin `<` conservan sus bytes anteriores.
No omite datos, no reescribe propuestas y no sustituye pesos ni modelo.
Los límites se aplican al JSON expandido; un exceso bloquea antes de generar,
sin recortar observaciones. Tres regresiones con motor fixture declarado
comprueban conservación, ausencia de IDs de control incrustados y límites;
la verificación pertinente aprobó 87 pruebas. No acreditan planificación aprendida.

El [recibo de lectura real del código del repositorio](autonomy-lossless-observations-real-read.json),
de las 11:32:33 UTC, SHA `4fc46cc18bf21780a602bac588ecdd1f5d2c1728a01607beb7bc1ea912d11b02`,
lee `contract.py` mediante `SafeFileTools.read_file`, verifica la lectura y
conserva el hash del original de 710 bytes con seis marcadores reservados.
El JSON anterior causaba `ConversationError` antes de generar; el nuevo
decodifica exactamente las observaciones y llega una vez al motor fixture,
cuya respuesta cruda permanece igual. Es evidencia real de IO y serialización
con fixture declarada; no ejecutó pesos propios ni entrenamiento.

## Comparación V2 de copia de argumentos

La [admisión del contraste](autonomy-action-copy-v2-enqueue.json) fijó dos brazos
desde el mismo ACTION 256 `e202eafc…`, configuración idéntica, 256 updates/600
segundos por brazo, uno tras otro. V2 diversifica nombres de archivo/informe;
V1 añade el mismo presupuesto de ensayo a los datos anteriores. En el snapshot
del 13 de octubre de 2026, 11:28:38 UTC, V2 terminó 256 en 157,4759767220021 s,
BEST 256 NLL 0,32269909729560214, propio `e70ea28e…`; genera DEV y el control
todavía no tiene estado nativo. Las tareas expuestas de uno/dos y el informe
no son corpus ni evaluación reservada. La comparación completa está pendiente.

La [evidencia propia V2](autonomy-demonstration/action-copy-v2-treatment-own-v1/evidence.json)
del 13 de octubre a las 11:27:59 UTC, SHA
`d4a6c742f05a517240712476dec1e277b0569e8fe3ebd5738199fafb3f143d52`,
ya ejecutó la misma tarea con temperatura cero, semilla 42 y máximo 128 tokens.
La primera propuesta real fue:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"dos.txt"},"reason":"Medir"}
```

El handler se ejecutó y su lectura se verificó independientemente. E202 había
propuesto `k.txt`, bloqueado antes del handler; ahora se observa copia correcta
en un caso expuesto. La segunda respuesta cruda, sin completarla ni corregirla,
fue:

```text
{"action":"tool","tool":"inspect_file","arguments":{"filename":"save_report","arguments":"filename":"spect_file","arguments"filename":"spect_file","save_file","arguments":"filename":"save_report","save_report",""
```

Alcanzó 128 tokens con prompt 933 sin descartes; terminó `blocked_model`, dos
pasos/un fallo, informe del modelo ausente y originales iguales. El objetivo
mínimo de 336 tokens también produjo JSON inválido hasta 128. La copia correcta
parcial no demuestra un plan generalizado ni trabajo completo. La fixture del
mismo paquete conserva su informe separado; no sustituye al modelo fallido.

Para repetir el diagnóstico con el [artefacto V2 completo](../../models/learning-repair-20261013/autonomy/action-copy-v2-treatment-final-best256/)
ya exportado y salida nueva, desde la raíz del repositorio:

```bash
python3 examples/demo_operational_autonomy.py \
  --model-dir models/learning-repair-20261013/autonomy/action-copy-v2-treatment-final-best256 \
  --output-dir private/autonomy-copy-v2-new
```

Separa la fixture explícita de las propuestas `own_model`. Repite las mismas
tareas, herramientas, schemas, límites, temperatura cero y semilla 42 con
el artefacto completo `models/learning-repair-20261013/autonomy/action-copy-v2-control-final-best256`,
siempre en otra carpeta nueva.
Conserva respuestas crudas y estado persistido, argumentos autorizados,
efectos/verificación, hashes originales e informe creado por el modelo.
Evalúa aparte la opción READ repair con el mismo valor para ambos brazos;
`examples/demo_autonomy_argument_repair.py` utiliza una reparación explícita.
Estos comandos son instrucciones, no nuevas ejecuciones registradas aquí.

La [comprobación adicional con dos tareas nuevas](action-copy-v2-development-tasks-v1/summary.json)
fijó candidatos, nombres y objetivos antes de la primera generación. Los cuatro
intentos reales pasaron por la API y el controlador, con reparaciones desactivadas:
ninguno ejecutó herramientas o creó su informe. Tratamiento y control obtuvieron
0/2 tareas completas. Los argumentos fuera de la lista del host y una propuesta
mal formada explican los bloqueos; conservar los originales no se cuenta como éxito.
El [helper congelado](../../examples/compare_action_copy_candidates.py) registra
propuestas literales, estado persistido y hashes. Su receta se reproduce en un
workspace nuevo, conservando las evidencias publicadas y sus exclusiones de TRAIN.

El [contraste operativo posterior](autonomy-action-copy-v2-real-matched-development.json),
de las 11:37:03 UTC, SHA `69f99163a165ae820f2f0c689b7aa2902e51727b4c62b5ea0368b6b0a046aae5`,
registra ambos brazos cerrados 256: tratamiento 157,475977 s/control 131,484277 s.
Comprueba igualdad del primer prompt, originales,schemas/contexto 2.048 y
decodificación 128 tokens/temperatura 0/semilla 42. El [control 734c real](autonomy-demonstration/action-copy-v2-control-own-v1/evidence.json)
propuso literalmente:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"v.txt"},"reason":"Medir"}
```

Terminó `blocked_policy`, un paso/un fallo, cero efectos, sin informe propio.
Su tarea mínima produjo JSON inválido, 82 tokens/EOS. Padre E202 también queda
bloqueado por `k.txt`; tratamiento E70 ejecuta una inspección autorizada de
`dos.txt`, verificada, y falla en el siguiente JSON 128. Todos conservan originales;
no se relajaron permisos ni se reparó la respuesta con otro modelo.
Se observa una selección de archivo correcta parcial frente al ensayo V1 en
esta tarea expuesta. No acredita generalización, plan completo ni conversación;
ninguno produjo el informe del modelo. Las NLL sobre datos distintos y las
latencias con evaluaciones concurrentes no se comparan como mejora del modelo.

JSON válido o una NLL menor no aceptan trabajo ni conversación: la aceptación
conversacional sigue en 36/40 completas, 8/10 diálogos y cero críticas/técnicas.
Los resultados anteriores 0/40 y 0/12 significados permanecen intactos.

## Reproducir y usar

La demostración crea una carpeta nueva y usa sólo archivos ficticios propios:

```bash
python examples/demo_operational_autonomy.py \
  --output-dir private/mi-demo-autonomia \
  --model-dir models/learning-repair-20261013/conversation-15m/intermediate-step004096
```

Genera `evidence.json`, los originales y el informe. Registra peticiones y
respuestas reales, las dos procedencias y cancelación. Guarda experiencias;
**no modifica ni reentrena pesos**.

Para habilitarlo en el servidor existente, configura `UNIDAD_API_TOKEN` con
una credencial privada y utiliza directorios existentes y separados:

```bash
python -m ia_propia.brain serve --body mi-aplicacion \
  --state-dir private/brain-autonomia \
  --conversation-model models/learning-repair-20261013/conversation-15m/intermediate-step004096 \
  --autonomy-input-root /ruta/entradas \
  --autonomy-report-root /ruta/informes \
  --autonomy-max-steps 8 --autonomy-max-seconds 60
```

Las entradas admitidas son archivos regulares capturados al iniciar; no admite
rutas arbitrarias ni enlaces simbólicos. Los informes no sobrescriben archivos.
La instalación atómica del informe usa Linux y `/proc/self/fd`.

Envía una petición Bearer autenticada a `/v1/requests`:

```json
{
  "schemaVersion": 1,
  "requestId": "analisis-1",
  "operation": "autonomy.start",
  "input": {
    "goal": "Analiza los archivos y guarda un informe sin modificarlos.",
    "task_inputs": {"files": ["uno.txt", "dos.txt"], "report": "informe.json"}
  }
}
```

Utiliza el `task_id` devuelto con `autonomy.step`, `autonomy.run`,
`autonomy.status`, `autonomy.cancel` o `autonomy.resume`. `run` es síncrono y
acotado; el servidor permite cancelar desde otra petición. La memoria permite
reanudar tras reiniciar el servidor con los mismos recursos y políticas.

Las integraciones de Python pueden registrar herramientas con `AutonomyTool`
y conectar `autonomy_tools`, `autonomy_limits`, `autonomy_completion_verifier`
y `autonomy_input_validator` a `Brain.connect`. Una afirmación `finish` del
modelo no basta: el verificador del host debe acreditar el objetivo. Para
herramientas sensibles, configura `autonomy_operator_token` y usa
`autonomy.approve` con el nonce pendiente y esa credencial. No expongas la
credencial al proponente. El registro inicial de archivos sólo utiliza lecturas
y escritura de informes nuevos; no registra comandos de shell o gastos.

Las herramientas propias con recursos o configuración mutable deben definir
un `policy_id` estable que cambie al cambiar esa configuración. El registro
incluido vincula raíces, archivos y límites a esa identidad; los nombres de
callbacks por sí solos no identifican los recursos de una integración nueva.

Verificación pertinente:

```bash
python -m unittest tests.test_brain_autonomy_controller \
  tests.test_brain_autonomy_api tests.test_autonomy_file_tools
```

Esas pruebas comprueban IO real, autenticación, permisos, aprobación, bloqueo
por contexto, cancelación concurrente, aislamiento, reanudación y evidencia.
La evaluación de conversación se conserva separadamente en
[comparison](comparison/README.md).
