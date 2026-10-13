# Usar SOVENGI 0.9

La etapa 0.9 entrenó un selector de coordinación de 18 parámetros sobre
candidatos originales de los 35 Transformers. Está conectado mediante
`configs/sovengi-coordination-candidate.json`. En la evaluación reservada quedó
en 2/12 cumplimientos literales, igual que las reglas anteriores; se conserva
como candidato experimental. El [informe de entrenamiento](sovengi-coordination-20261013/README.md)
identifica los pesos y separa la comprobación literal de calidad conversacional.

```bash
sovengi --config configs/sovengi-coordination-candidate.json models
sovengi --config configs/sovengi-coordination-candidate.json chat
```

La [integración conversacional de NERVA](nerva-conversation-20261013/README.md)
se aplica también a esta entrada: `brain.conversation_decoder` en la configuración
común selecciona `"torch"` o `"nerva"`, y `brain.conversation_cache` controla la
caché por respuesta. Torch con caché queda predeterminado por la medición C7;
NERVA requiere Node.js 18+. Cada miembro conserva sus propios pesos.
`sovengi chat` y `sovengi ask` admiten `--seed`, `--top-k`, `--top-p` y
`--repetition-penalty` con el mismo contrato de la API. Los controles de variedad
y repetición son neutrales por defecto y no acreditan nuevo aprendizaje.

SOVENGI conecta un catálogo de **35 Transformers propios** mediante
`ConversationEnsemble` y el mismo `ConversationEngine` que utiliza cada participante.
El coordinador se comparte entre productos, conversación y propuestas de
herramientas. Sus participantes son generadores con pesos independientes;
sus nombres y etapas no acreditan nuevas capacidades de razonamiento.
La respuesta entregada es el texto original de un candidato seleccionado.

El [modelo independiente de contexto](context-model-20261013/README.md) está
conectado antes de ese motor. Selecciona texto original del historial con pesos
propios, conserva el sistema y la petición actual y reserva tokens para generar.
Los turnos antiguos se recuperan desde SQLite; las respuestas anteriores se
mantienen con rol `assistant`, sin aceptarlas automáticamente como hechos.

`sovengi --context-session trabajo chat` permite recuperar el mismo contexto
local al reabrir. El identificador lo elige el anfitrión y no cambia carpetas ni
herramientas. `/nuevo` elimina el contexto de esa conversación y conserva las
notas explícitas. `sovengi --no-context chat` permite comparar la misma
coordinación sin el selector. Si las instrucciones y el mensaje actual no caben con el presupuesto
de respuesta, devuelve `context_budget` y pide dividir el texto; no los recorta
silenciosamente. Cada participante mide esa entrada con su propio tokenizador
y conserva el mensaje de sistema y la petición actual. Para ajustarla puede
retirar turnos antiguos completos, dejando constancia. Un miembro cuyo contexto
mínimo no cabe se omite con `minimum_context_exceeds_capacity`.

## Empezar

Desde la raíz del repositorio completo, con Python 3.11 o posterior:

```bash
python -m pip install -e '.[conversation]'
sovengi chat
sovengi ask 'Hola, ¿qué puedes hacer?' --max-new-tokens 96
sovengi status
sovengi models
sovengi score 'El taller abre el jueves.'
```

`python -m ia_propia.sovengi` y `unidad sovengi` son entradas equivalentes.
`status` muestra servicios y presupuestos; `models` incluye identidades,
disponibilidad y estados `not_loaded`, `resident`, `evicted`, `unavailable` o `failed`.
Los pesos se cargan al usarlos, con comprobación de hashes y arquitectura.
La referencia primaria configurada tiene que estar instalada para abrir la aplicación.
Un miembro ausente o rechazado permanece visible en el catálogo y en la traza.
Si no se alcanzan dos generaciones válidas, la petición falla explícitamente;
también falla cuando todos sus candidatos están vacíos. Estos estados validan
la ejecución, no la calidad conversacional.

## Una configuración y 35 modelos coordinados

`configs/gnosis.json` fija la coordinación; su campo
`brain.conversation_ensemble.catalog` apunta al
[catálogo completo](../configs/conversation-models.json). `name`, `modelId`,
SHA de pesos y tokenizador, arquitectura y procedencia identifican cada miembro.
Los siguientes prefijos remiten a identidades completas de ese catálogo:

| Referencia | Parámetros | Contexto / vocabulario | Prefijo de identidad |
| --- | ---: | --- | --- |
| `reference-wiki1` | 15.769.344 | 2.048 / 2.048 | `c0ee5f24…` |
| `memory-candidate` | 15.769.344 | 2.048 / 2.048 | `a52997f8…` |
| `C7` | 6.983.937 | 512 / 8.192 | `bd1975c2…` |
| `V7` | 6.967.296 | 512 / 8.192 | `9081da24…` |

Las dos primeras referencias tienen ocho capas y dimensión 384; C7 y V7,
seis capas y dimensión 256. Hay cinco configuraciones de arquitectura y tres
tokenizadores distintos en el catálogo. Sus pesos proceden de inicialización
aleatoria propia o continuaciones de ese linaje. Todos conservan
`productionAccepted: false`.

En la configuración habitual, `reference-wiki1` y `memory-candidate` ocupan
dos puestos estables; los otros dos participantes rotan entre los 33 restantes.
La rotación pertenece a la instancia del coordinador: conserva la aplicación
para continuarla; arrancar otra instancia la reinicia. Los participantes
indisponibles se registran como omitidos. Hasta cuatro modelos permanecen
residentes en una caché LRU; un modelo retirado puede volver a cargarse cuando
le corresponda. Los cuatro puestos por respuesta no significan que los 35
estén simultáneamente en RAM.

El presupuesto habitual admite hasta cuatro candidatos, exige al menos dos
generaciones válidas y reparte **256 tokens nuevos** en total. El límite pedido
con `--max-new-tokens` también limita cada candidato. Hasta dos participantes
evalúan los candidatos, con un máximo conjunto de ocho llamadas de puntuación.
Cada respuesta evaluada puede ocupar hasta 256 tokens en el tokenizador de su
evaluador; la configuración conserva cero tokens de puntuación con ventana
deslizante. Cada evaluador tokeniza el texto original, reserva su respuesta y
mantiene el sistema y la petición actual. La selección combina las posiciones
obtenidas dentro de cada evaluador completo y el soporte léxico de las fuentes.
Ese soporte no demuestra que una afirmación sea correcta.

El límite de **60 segundos** es cooperativo y se comprueba entre operaciones.
Una carga o inferencia en curso puede terminar después: no hay interrupción
forzosa. La salida `ensemble` registra participantes e identidades ejecutadas,
evaluadores completos, omisiones y fallos, tokens cobrados, tiempo real, exceso
de presupuesto, cargas, retiradas y `selectedModelId`. El `modelId` de la
respuesta corresponde al generador elegido; la identidad primaria sigue
identificando la configuración del coordinador. La traza no incluye una
segunda copia de los textos candidatos.

Un checkout parcial de CI puede incluir sólo algunos artefactos. Los ausentes
permanecen listados y no se descargan durante una petición. La entrega completa
incluye 35 archivos de pesos distintos; el
[recibo de catálogo](sovengi-ensemble-20261013/model-catalog.json) verifica
**1.417.104.076 bytes** por SHA-256 y describe las referencias duplicadas.
El SHA-256 de `configs/conversation-models.json` es
`cafd1650de402d8b5365123952401863478c85b654e56d2d75c6908bbc180ffe`.
El delta de copia propio se entrega además como un artefacto completo de
**27.943.524 bytes**, reproducido con su script original; su
[recibo](sovengi-ensemble-20261013/delta-reconstruction.json) identifica padre,
delta y resultado. Su SHA de pesos es
`c3d2d09bd4f82efd8c92f649482441e1e40bc29cd192a35ddfc1d56a45df400b`.
La carga funciona sin red ni reconstrucción manual.

La [guía de coordinación](sovengi-ensemble-20261013/README.md) reúne decisiones
y verificaciones. El [benchmark](sovengi-ensemble-20261013/benchmark.json) usa
pequeños modelos propios aleatorios para comprobar la caché y la conservación
de resultados; no mide conversación ni rendimiento del catálogo completo.
La [evaluación real](sovengi-ensemble-20261013/real-evaluation.json) conserva
respuestas y trazas de los pesos publicados, con sus fallos y límites.
Su comprobación de cobertura ejecutó los **35 modelos en 17 peticiones**,
con dos tokens por candidato y un máximo de cuatro residentes. La comparación
conversacional generó 16 respuestas con un tope de 24 tokens; no acredita calidad
aceptada. En el benchmark separado, cuatro modelos aleatorios pequeños pasaron
de **47,14 a 24,31 ms** de mediana al conservar cuatro residentes en vez de uno,
manteniendo el ganador y los evaluadores. Ese resultado no proyecta tiempos
de los modelos publicados ni de otras máquinas.

La [etapa de memoria del 13 de octubre](sovengi-memory-20261013/README.md) sí
entrenó pesos nuevos de esa línea. Para probar el candidato con la misma consola,
memoria, API y motor:

```bash
sovengi --config configs/sovengi-memory-candidate.json chat
```

Ese comando cambia la primaria a `memory-candidate` dentro de la misma
coordinación y mantiene separado su estado privado. El control de memoria pasó
de 1/25 a 22/25 respuestas exactas en ejercicios acotados; la conversación
general sigue sin aceptación. Abrir cualquiera de estas configuraciones no
inicia entrenamiento.

| Pieza | Función en SOVENGI |
| --- | --- |
| Brain + ConversationEnsemble + ConversationEngine | Coordinación y generación compartidas entre consola, Python y HTTP. |
| Memoria | Conserva texto guardado expresamente y lo recupera en el diálogo. |
| Modelo de contexto | Ordena fuentes del historial, prepara una entrada acotada y conserva procedencia. |
| Herramientas y controlador | Ejecutan tareas acotadas con observaciones, permisos y límites. |
| GNOSIS | Selecciona los pesos y configura etapas de aprendizaje. |
| NERVA | Conserva el backend que esté implementado y verificado en GNOSIS; no es otro modelo con conocimientos propios. |
| Perfil 15m | Describe la arquitectura de la línea elegida. Los perfiles 50m, 100m y 1b son recetas, no modelos ya entrenados. |
| C7 y candidatos anteriores | Participantes del catálogo y referencias conservadas para comparar. |
| RNN de caracteres y MLP de métricas | Diagnóstico de candidatos y pronóstico conectado explícitamente a conversación. |

## Modelos auxiliares y observaciones

La RNN de caracteres conserva sus **2.903 parámetros** propios. `language.score`
y `sovengi score` devuelven pérdida y proporción de caracteres desconocidos,
con un máximo de 512 caracteres. El coordinador emplea esos valores como
diagnóstico y desempate. Esa puntuación no comprueba hechos ni comprensión
del español. `language.generate` conserva la generación de caracteres mediante
el backend NumPy o NERVA elegido por el anfitrión.

El MLP de **49 parámetros** conserva el dominio de entrenamiento:
`source=wikimedia:en.wikipedia.org` y `metric=views`. `attention.forecast`
pronostica a partir de observaciones explícitas con procedencia. La operación
`conversation.forecast` conecta ese pronóstico, el gestor de contexto y la
coordinación conversacional, con `predictionNotObservedFact: true` y la
evidencia numérica original. El modelo conversacional todavía puede explicar
mal esa evidencia; el pronóstico no se presenta como un hecho observado.

Para usar las observaciones históricas de ejemplo, desde la raíz:

```bash
sovengi forecast 'Resume el pronóstico de vistas y sus límites.' \
  --observations examples/attention-observations.json \
  --signal-id 'wikimedia:en.wikipedia.org:League_of_Legends:_Wild_Rift' \
  --max-new-tokens 64
```

El archivo contiene observaciones de octubre de 2024; no proporciona vistas
en directo. Las peticiones aportan las observaciones y el núcleo valida su
fuente y métrica. La [auditoría de conexiones](sovengi-connectivity-20261013/audit.json)
explica qué componentes participan en conversación y qué operaciones son
explícitas o corresponden al aprendizaje.

## Memoria privada

```bash
sovengi remember 'El taller abre el jueves.'
sovengi memory taller
sovengi ask '¿Cuándo abre el taller?'
```

En el chat sirven `/recordar texto`, `/memoria palabra`, `/nuevo` y `/salir`.
Las notas se guardan mediante esas operaciones explícitas. Con el gestor
habilitado, el historial real se conserva además en SQLite privado por
conversación; `--context-session` permite recuperar el mismo identificador.
Reiniciar la conversación no borra las notas. El historial, las notas y las
respuestas generadas no se convierten automáticamente en corpus ni cambian pesos.

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

La misma coordinación propone acciones y el controlador las comprueba. El trabajo
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
delega al entrenador reanudable de GNOSIS. El ciclo completo está disponible:

```bash
sovengi plan --stage conversation
sovengi learning-status --stage conversation
# Con corpus e identidad configurados y presupuesto explícito:
sovengi train --stage conversation --budget-seconds 60
sovengi learning-status --stage conversation
sovengi export --stage conversation
```

`export` escribe el BEST disponible de la etapa en su carpeta `artifact`;
requiere un checkpoint real. Los corpus continúan sin configurarse
en las recetas de uso: no se asignan datos personales. El experimento terminado
conserva su corpus y configuración propios; su presupuesto ya se consumió.
Una continuación requiere otra etapa finita declarada. La existencia de recibos
anteriores del supervisor no acredita que siga activo en otro entorno.

Una corrida nueva no cambia el modelo servido hasta evaluar sus respuestas y
actualizar explícitamente ruta e identidad. La conversación general, el uso
aprendido de memoria y la planificación autónoma siguen siendo experimentales.
