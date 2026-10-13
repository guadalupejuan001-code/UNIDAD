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
