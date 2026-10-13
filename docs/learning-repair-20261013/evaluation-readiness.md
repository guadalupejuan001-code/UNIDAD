# Preparación de la revisión conversacional

Esta nota corresponde a una inspección estática del 13 de octubre de 2026. No
ejecuta inferencia, entrenamiento ni una batería de software. La etapa SFT
`conversation-15m` continúa bajo su supervisor. No se ha abierto el contenido
de la nueva suite reservada ni se han utilizado sus casos para esta nota.

La revisión que sigue será de IA del proyecto. Una segunda revisión de otro
agente conserva independencia de juicio, pero no constituye evaluación humana
independiente ni valida la exactitud de respuestas mediante otra IA en inferencia.

## Contrato comprobado por lectura

[`run_learning_evaluation.py`](../../examples/run_learning_evaluation.py)
carga cada artefacto mediante `ConversationEngine`, que verifica configuración,
tokenizador, pesos y su identidad mediante el cargador común. La referencia es
[`conversation-reference-c7`](../../models/conversation-reference-c7/README.md):

- Modelo: `unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2`.
- Pesos: SHA-256 `666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836`.
- Tokenizador: SHA-256 `fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d`.
- Configuración: SHA-256 `d30b5cd427fb9711d58e8eb55daf9a32529f872248b706805001e9eebd02f12b`.

La comparación prospectiva usa las mismas cuarenta peticiones, temperatura cero,
semilla 42, hasta 160 tokens nuevos por respuesta y un hilo de PyTorch. Cada
diálogo empieza con historia vacía; después de cada petición se añade únicamente
el texto que ese modelo generó. No se insertan respuestas esperadas ni se corrige
la historia. El sistema predeterminado se añade mediante `prepare_messages`.
La generación admite texto y EOS, y excluye PAD, BOS y marcadores de rol como
tokens generados. Un EOS inmediato produce respuesta vacía; el runner conserva
el fallo y bloquea los turnos posteriores de ese diálogo, sin inventar respuestas.

La plantilla explícita es BOS, marcador de rol, contenido y EOS por mensaje,
seguida del marcador de asistente. La máscara causal y la supervisión de última
respuesta pertenecen al código de entrenamiento; no son un sustituto de revisar
la conversación libre. En esta etapa se mantiene `assistant_last_reply`, con
reducción `sequence_mean` y todos los parámetros entrenables.

El candidato SFT conserva el tokenizador propio de su base de 15.179.520
parámetros; su BPE tiene 2.048 entradas. C7 tiene 6.983.937 parámetros, BPE de
8.192 entradas y cabeza de copia. Ambos usan contexto de 512 tokens. Compararlos
por respuestas mide el cambio de paquete completo —base, capacidad, datos y
ajuste—; no identifica por sí solo la contribución causal de una variable.
Las mismas palabras pueden consumir cantidades distintas de tokens. Se conservan
por artefacto los tokens efectivos y cualquier recorte.

El hook del supervisor genera esta suite de desarrollo después del cierre SFT,
con un hilo, 300 segundos, hasta 6.144 MiB de RSS y 512 MiB de salida. No juzga,
promueve ni certifica el modelo. Un hook incompleto es evidencia de ejecución
incompleta, nunca cuarenta respuestas evaluadas.

## Suite expuesta y cobertura

La suite `learning-repair-20261013-development-v2` consta de diez diálogos de
cuatro turnos y 38 condiciones críticas. El sello fija:

| Archivo | SHA-256 |
| --- | --- |
| `cases.json` | `4a657c92a7ab39eb55bca0df95a70ed88c567ab5bcc7f83b074582f08ad8dfc2` |
| `rubric.json` | `95ed7bb9ffe017858980865576a411e4c75ef6bea2acde400dcfe18e36337ed3` |
| `freeze.json` | `93f19c921cb6022eb2dba5bb5e734849cafad331693478b5ca606b690f311026` |

| Diálogo | Hechos y cambios que deben conservarse |
| --- | --- |
| Biblioteca | Viernes a las seis cambia a jueves; hermana cambia a padre; viernes por la mañana ya no cumple el plazo. |
| Cine | Dos adultos y un niño: 18 euros; tarifa del chico corregida: 21; se retira un adulto: dos entradas, 14 euros; jueves a las ocho y media se conserva. |
| Grifo | Primero sólo caliente de cocina; luego ambas salidas de cocina; baño normal; lunes de nueve a once; vecino cambia a madre. |
| Aviso laboral | Mañana, diez cambia a once; sala grande cambia a videollamada; quitar enlace y mantener una frase sin saludo; redactar no equivale a enviar. |
| Invitación | Domingo cambia a sábado; cinco y pista se conservan; baloncesto cambia a paseo; una frase, «quien se anime», sin emojis. |
| Consulta | Abuelo explica dolor; usuario aporta lista; hermana conduce y espera fuera; abuelo termina entrando solo; una hipótesis no cambia los papeles. |
| Gotas | Humedad del aire y superficie fría; no agua atravesando el cristal; vaso vacío también puede formar gotas; una frase sin «condensación». |
| Paquete | Seguimiento no demuestra robo; vecina sólo ve furgoneta; foto en puerta azul frente a puerta verde; repartidor confirma número vecino, sin recuperación aún. |
| Charla tranquila | Seguir la charla sin consejo; lluvia cambia a cafetería casi vacía; conservar preferencia de hoy sin inventar vivencias. |
| Nota | Taller miércoles cambia a martes; contexto no implica entrenamiento ni persistencia; sin herramienta o contacto no hay envío; dejar texto para copiar. |

Todas las peticiones cubren respuesta actual y español. Hay 31 turnos etiquetados
con continuidad, 22 con hechos, 15 con correcciones, siete con papeles, cinco con
límites de herramientas, cinco con explicación, cinco con incertidumbre, cuatro
con preferencias y cuatro con aritmética cotidiana. La aritmética se informa
también por separado para no atribuir sus fallos exclusivamente al español.
Las etiquetas se solapan y no son cuarenta medidas independientes por habilidad.

Esta suite comprueba papeles de personas dentro de una narración de un usuario.
No demuestra identificación general de varios usuarios reales, cobertura de todos
los nombres nuevos ni todo el español. Tampoco prueba herramientas, autonomía,
memoria persistente o actualización de parámetros provocada por una charla.
Los casos quedan expuestos para desarrollo; no son corpus, filtro de corpus ni
una prueba reservada nueva.

## Protocolo de revisión de las salidas reales

1. Fijar rutas e identidades de C7 y del candidato, SHA-256 de los cuatro archivos
   de artefacto, sello de la suite y configuración de generación. Crear directorios
   de salida nuevos; conservar intentos incompletos. Antes de abrir el reservado,
   registrar la selección del candidato y la configuración definitivas.
2. Verificar que cada turno contiene la petición congelada y las respuestas
   anteriores realmente generadas por ese mismo modelo. Leer cada diálogo entero
   en su orden; los errores del asistente son contexto del siguiente turno, no
   nuevos hechos que anulan una corrección del usuario.
3. Revisar por turno `relevance`, `contextGrounding` y `spanish`; revisar
   `criticalCorrect` en los 38 turnos críticos. Registrar grado completo, parcial
   o fallido y una evidencia breve. No exigir literalidad del texto esperado.
   Una respuesta parcial debe aportar contenido correcto que responda a lo
   solicitado: eco, repetición de nombres y apoyo genérico no bastan.
4. Marcar como crítica una afirmación identificable incompatible con los hechos,
   los papeles o los límites del turno. Citar la frase concreta y el dato
   contradicho. Omitir la respuesta, repetir texto o producir una frase
   incomprensible puede ser fallido sin ser automáticamente crítico. Un mero
   incumplimiento de formato no se convierte en invención factual.
5. Registrar separadamente excepción, vacío, bloqueo, recorte inicial de contexto
   y `stopReason=token_budget`. Conservar respuesta, tokens y error sin editar.
   Un texto cortado se revisa como se recibió; no se completa mentalmente ni
   aumentando el presupuesto después de conocer su resultado.
6. Una segunda persona o agente revisa las mismas salidas sin leer los juicios
   primarios. Conservar ambos archivos, metodología, desacuerdos y una eventual
   adjudicación en un archivo nuevo. Los revisores de IA se identifican como tal.
7. Informar completas, parciales y fallidas, diálogos de cuatro turnos completos,
   críticas y técnicas; transiciones de mejora y deterioro en los mismos turnos;
   instrucciones, continuidad, correcciones, claridad y repetición. Informar
   latencia y RSS del proceso evaluador, separadas de las del entrenador.

La aceptación exige al menos 36 completas de 40, ocho diálogos completos de diez
y cero fallos críticos o técnicos. Cumplirlo en esta suite expuesta sólo acredita
esta prueba básica de desarrollo. La aceptación final exige una suite nueva
reservada y un candidato fijado; ninguna de las dos acredita competitividad
comercial por sí sola. NLL menor, más pasos, archivos de pesos o CI verde no
cambian estos criterios.

## Matices del runner que deben conservarse en el informe

En la versión congelada del runner, `assess` agrega los fallos técnicos y exige
que sean cero para `accepted`. Sin embargo, el contador `fullyMetTurns` se
calcula a partir de las dimensiones semánticas y de `criticalCorrect`, sin
excluir automáticamente un turno técnico; lo mismo ocurre con los diálogos.
Por ejemplo, una respuesta semánticamente válida que agota 160 tokens podría
sumar en ese contador y aun así causar `accepted=false`. No presentar ese
contador como completas según la rúbrica sin comprobarlo. Conservar el resultado
del runner y añadir un conteo independiente donde cualquier fallo técnico
impide el grado completo y el diálogo completo. No modificar el runner
congelado ni los resultados para esconder la diferencia. No poner una dimensión
semántica a falso sólo para compensar ese contador: cada dimensión conserva su
juicio real y el fallo técnico queda en su propio campo.

`discardedPromptTokens` mide el recorte al preparar el prompt. Durante cada
respuesta, `chat` vuelve a tomar los últimos 512 tokens; puede desplazar contexto
adicional si la respuesta crece. A partir de una salida completa puede
reconstruirse el último contexto de generación concatenando
`effectiveContext.tokenIds` con `generatedTokenIds` salvo el último, y tomando
los últimos 512. La longitud antes de ese recorte es
`promptTokens + generatedTokens - 1`; cuando supera 512 hay desplazamiento
durante la generación, aunque el recorte inicial registrado sea cero. Registrar
qué información necesaria quedó fuera antes de atribuir el fallo a aprendizaje.

El runner actual penaliza cualquier recorte inicial, incluso antes de juzgar
cuáles de sus tokens eran necesarios, y registra todos los finales por
presupuesto como técnicos. Se conservan ambas reglas al comparar. La futura
revisión puede precisar su efecto con los tokens reales, sin alterar la suite
ni liberar retrospectivamente errores.

## Lo que permite afirmar la evidencia histórica

En las salidas públicas anteriores `development-c7`, `development-v7` y
`development-history-generated384`, los cuarenta prompts de cada corrida
registran cero tokens descartados y cero excepciones. Las máximas longitudes
prompt más continuación fueron respectivamente 223, 397 y 211, por debajo de
512; no aparece desplazamiento durante generación. C7 y `generated384`
terminaron sus cuarenta respuestas por EOS; v7 agotó el presupuesto una vez.

Por tanto, el fallo común histórico no se explica por pérdida de ventana de
contexto. C7 contestó «En la cafetería, el total está en la cafetería» a una
pregunta que distinguía total y precio unitario, y «No cambió el total» ante una
corrección explícita de ocho a nueve euros. Esos fallos aparecen con los hechos
visibles, roles explícitos y presupuesto disponible. La evidencia favorece
investigar aprendizaje lingüístico, atención útil a la petición y correcciones;
no demuestra por sí sola cuál de capacidad, exposición o composición de datos
es la causa. Los errores ocurren también en primeros turnos, donde no existe
historia generada previa que pueda explicarlos.

El nuevo SFT todavía requiere generación libre sobre esta suite. No extrapolar
su resultado desde el descenso de NLL ni asumir que la nueva base eliminó
aquellos fallos. Esta nota queda lista para acompañar la revisión cuando existan
las cuarenta respuestas reales de cada artefacto.
