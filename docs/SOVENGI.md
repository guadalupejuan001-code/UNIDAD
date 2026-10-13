# Usar SOVENGI

La [integración conversacional de NERVA](nerva-conversation-20261013/README.md)
se aplica también a esta entrada: `brain.conversation_decoder` en la configuración
común selecciona `"torch"` o `"nerva"`, y `brain.conversation_cache` controla la
caché por respuesta. Torch con caché queda predeterminado por la medición C7;
NERVA requiere Node.js 18+. Ambos conservan el modelo propio seleccionado.
`sovengi chat` y `sovengi ask` admiten `--seed`, `--top-k`, `--top-p` y
`--repetition-penalty` con el mismo contrato de la API. Los controles de variedad
y repetición son neutrales por defecto y no acreditan nuevo aprendizaje.

SOVENGI une los servicios que ya existen. Una instancia carga un solo motor
conversacional y comparte esos pesos entre sus productos, conversación y
propuestas de herramientas. Las respuestas proceden del modelo propio; no hay
respuestas de otra IA ni frases programadas para simular sus capacidades.

El [modelo independiente de contexto](context-model-20261013/README.md) está
conectado antes de ese motor. Selecciona texto original del historial con pesos
propios, conserva el sistema y la petición actual y reserva tokens para generar.
Los turnos antiguos se recuperan desde SQLite; las respuestas anteriores se
mantienen con rol `assistant`, sin aceptarlas automáticamente como hechos.

`sovengi --context-session trabajo chat` permite recuperar el mismo contexto
local al reabrir. El identificador lo elige el anfitrión y no cambia carpetas ni
herramientas. `/nuevo` elimina el contexto de esa conversación y conserva las
notas explícitas. `sovengi --no-context chat` sirve para comparar el mismo
generador. Si las instrucciones y el mensaje actual no caben con el presupuesto
de respuesta, devuelve `context_budget` y pide dividir el texto; no los recorta
silenciosamente. La ventana aprendida del Transformer continúa siendo 2.048.

## Empezar

Desde la raíz del repositorio completo, con Python 3.11 o posterior:

```bash
python -m pip install -e '.[conversation]'
sovengi chat
sovengi ask 'Hola, ¿qué puedes hacer?' --max-new-tokens 96
sovengi status
```

`python -m ia_propia.sovengi` y `unidad sovengi` son entradas equivalentes.
Si falta un peso o no coincide su identidad, el arranque falla: no cambia a otro
modelo silenciosamente. `status` muestra el modelo realmente cargado y los
servicios disponibles. Su éxito valida la carga, no la calidad conversacional.

## Una configuración y un modelo elegido

`configs/gnosis.json` sigue siendo la configuración común. El modelo general
publicado seleccionado es `models/learning-repair-20261013/pattern-relations-v1/reference-wiki1`:
15.769.344 parámetros, ocho capas, contexto 2.048 y vocabulario 2.048. Su
`expected_model_id` fija los pesos; el cargador verifica los hashes del artefacto.
SOVENGI es el nombre del producto en el prompt y la interfaz, no un nuevo modelo
entrenado por cambiar una etiqueta.

La [etapa de memoria del 13 de octubre](sovengi-memory-20261013/README.md) sí
entrenó pesos nuevos de esa línea. Para probar el candidato con la misma consola,
memoria, API y motor:

```bash
sovengi --config configs/sovengi-memory-candidate.json chat
```

Su identidad está fijada en esa configuración y su estado privado está separado
del de la referencia. El control pasó de 1/25 a 22/25 respuestas exactas, pero
la conversación general sigue sin aceptación. `configs/gnosis.json` conserva la
referencia anterior para comparar. Ninguna de estas configuraciones inicia
entrenamiento al abrir el chat.

| Pieza | Función en SOVENGI |
| --- | --- |
| Brain + ConversationEngine | Una implementación de generación y protocolo para consola, Python y HTTP. |
| Memoria | Conserva texto guardado expresamente y lo recupera en el diálogo. |
| Modelo de contexto | Ordena fuentes del historial, prepara una entrada acotada y conserva procedencia. |
| Herramientas y controlador | Ejecutan tareas acotadas con observaciones, permisos y límites. |
| GNOSIS | Selecciona los pesos y configura etapas de aprendizaje. |
| NERVA | Conserva el backend que esté implementado y verificado en GNOSIS; no es otro modelo con conocimientos propios. |
| Perfil 15m | Describe la arquitectura de la línea elegida. Los perfiles 50m, 100m y 1b son recetas, no modelos ya entrenados. |
| C7 y candidatos anteriores | Referencias para comparar calidad y poder volver atrás. Sus pesos no se mezclan. |
| Modelos pequeños de caracteres y métricas | Servicios auxiliares explícitos del núcleo; no contestan el chat de SOVENGI. |

## Memoria privada

```bash
sovengi remember 'El taller abre el jueves.'
sovengi memory taller
sovengi ask '¿Cuándo abre el taller?'
```

En el chat sirven `/recordar texto`, `/memoria palabra`, `/nuevo` y `/salir`.
Sólo se guardan recuerdos mediante operaciones explícitas. El diálogo de la
consola permanece en memoria temporal; reiniciar la conversación no borra notas.
No se entrena automáticamente con notas ni con respuestas generadas.

La recuperación es por palabras, no comprensión semántica: selecciona hasta
tres tareas y 2.048 bytes, conserva campos completos y correcciones, excluye
intentos no aprobados y cita los eventos originales en `memoryContext`.
Si los recuerdos no caben junto a la conversación y la respuesta reservada,
se omiten. El historial por sí mismo puede exceder la ventana; la salida informa
`discardedPromptTokens`. El modelo todavía puede ignorar o interpretar mal un
recuerdo: esta integración no demuestra recuerdo fiable aprendido.

`--body` selecciona la memoria privada desde el anfitrión. Dos productos tienen
memorias separadas; Python permite compartirlas con `memory_scope` explícito.
Las peticiones no pueden cambiar rutas, pesos ni herramientas. El núcleo Brain
sin SOVENGI mantiene su contrato de contexto explícito.

## Herramientas y API

Las herramientas de archivos son opcionales y se habilitan con carpetas
existentes. Sólo admiten los archivos capturados al arrancar; para incorporar
otros se reinicia el producto. No sobreescriben originales ni informes.

```bash
mkdir -p private/sovengi/inputs private/sovengi/reports
# Coloca antes los archivos autorizados en inputs.
sovengi --input-root private/sovengi/inputs --report-root private/sovengi/reports work \
  'Inspecciona muestra.txt y guarda su informe' --files muestra.txt --report informe.json
```

El mismo modelo propone acciones y el controlador las comprueba. El trabajo
puede quedar bloqueado si la propuesta es inválida; no se sustituye por un plan
programado para aparentar éxito. `request` permite consultar, cancelar o reanudar
tareas con el protocolo de Brain. Las aprobaciones de operador requieren la
credencial configurada mediante `SOVENGI_OPERATOR_TOKEN`.

Configura `SOVENGI_API_TOKEN` con al menos 32 caracteres y ejecuta:

```bash
sovengi serve --host 127.0.0.1 --port 8765
```

La API usa `Authorization: Bearer ...`: `GET /v1/capabilities` y
`POST /v1/requests`. Por ejemplo, el cuerpo de una petición de conversación es:

```json
{"schemaVersion":1,"operation":"conversation.reply","input":{"messages":[{"role":"user","content":"Hola"}],"max_new_tokens":96,"temperature":0}}
```

Las integraciones Python usan `Sovengi(config).connect(body_id, tools=...)` y
`session.request(...)`. Deben conservar esa instancia y llamar a `close()` al
cerrar para liberar descriptores de herramientas.

## Qué falta aprender

No hay que entrenar todos los modelos: se continúa un candidato propio y se
compara contra las referencias con datos nuevos separados de entrenamiento.
`sovengi plan` muestra recursos y bloqueos; `sovengi train --budget-seconds 60`
delega al entrenador reanudable de GNOSIS. Los corpus continúan sin configurarse
en las recetas de uso: no se asignan datos personales. El experimento terminado
conserva su corpus y configuración propios; su presupuesto ya se consumió.
Una continuación requiere otra etapa finita declarada. La existencia de recibos
anteriores del supervisor no acredita que siga activo en otro entorno.

Una corrida nueva no cambia el modelo servido hasta evaluar sus respuestas y
actualizar explícitamente ruta e identidad. La conversación general, el uso
aprendido de memoria y la planificación autónoma siguen siendo experimentales.
