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

Para habilitarlo en el servidor existente, configura `UNIDAD_BRAIN_TOKEN` con
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

Verificación pertinente:

```bash
python -m unittest tests.test_brain_autonomy_controller \
  tests.test_brain_autonomy_api tests.test_autonomy_file_tools
```

Esas pruebas comprueban IO real, autenticación, permisos, aprobación, bloqueo
por contexto, cancelación concurrente, aislamiento, reanudación y evidencia.
La evaluación de conversación se conserva separadamente en
[comparison](comparison/README.md).
