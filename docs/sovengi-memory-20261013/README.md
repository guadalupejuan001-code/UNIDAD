# Entrenamiento de SOVENGI: memoria recuperada y conversación

La etapa terminó el 13 de octubre de 2026. Partió del modelo propio que carga
SOVENGI en `main` `2d989a5` para aprender a responder con hechos recuperados y
correcciones, manteniendo repaso conversacional. La integración ya recuperaba
la nota correcta, pero el modelo podía contestar con texto incorrecto.

## Resultado comprobado

| Control de desarrollo | Antes | Después |
| --- | ---: | ---: |
| Notas de apertura, 10 entidades reservadas y 2 hechos alternativos por entidad | 0/20 exactas | 20/20 exactas |
| Otros cinco turnos: ubicación, pertenencia, cita, corrección e información ausente | 1/5 exactas | 2/5 exactas |
| Total, misma decodificación y fin normal de respuesta | 1/25 exactas | 22/25 exactas |

Se conservan [todas las respuestas anteriores](before.json),
[todas las nuevas](after.json), el [recibo del entrenador](summary.json) y la
[revisión de resultados](review.json). El control de notas verifica combinaciones
reservadas dentro de una misma familia de ejercicios; no mide conversación
abierta ni situaciones arbitrarias de memoria. No se consultó el TEST final.

Ejemplo real: ante «¿En qué día de la semana abre el taller Lucerna?» con una
nota que dice jueves, la referencia repetía «El taller Lucerna abre el taller
de la persona…». El candidato generó «El taller Lucerna abre el jueves.».
Cuando la nota cambió a lunes, respondió lunes. Las salidas se generaron con los
pesos nuevos mediante SOVENGI; no se insertaron respuestas esperadas en el código
de inferencia.

Hay límites visibles: «La cita es una cita con un lunes.» es poco natural y
«¿Cuál es tu vecino?» no reconoce claramente que falta su nombre. La corrección
«El viernes.» sí responde al cambio de fecha, aunque no coincide literalmente
con la frase esperada. La revisión semántica es de Codex, sin evaluación humana
independiente; no se presenta el recuento exacto como porcentaje de inteligencia.

La corrida ejecutó **27 actualizaciones de los 15.769.344 parámetros** y consumió
**921,32 segundos** dentro del entrenador. Se detuvo al agotar el presupuesto
de 900 segundos, comprobado entre actualizaciones y seguido de validación y
guardado finales; no llegó al máximo de 512 pasos. El tiempo de entrenamiento,
exportación y evaluación posterior fue 927,81 segundos. El pico del proceso fue
775.512 KiB. La pérdida de validación pasó de 1,320885 a 0,808338; esa reducción
no sustituye la lectura de las respuestas. Se conservaron 482 registros TRAIN
y 58 de validación: 352 ficticios nuevos y 188 de repaso sin modificar.

El candidato está en
[`models/sovengi-memory-20261013/candidate-best`](../../models/sovengi-memory-20261013/candidate-best),
con identidad `unidad-conversation-a52997f88799597080c93ec3e79ba4029d9eb5b6d756e2136098dd16e909ca0c`.
Sus pesos tienen SHA-256
`08a4f57c420bc3c454abc326f61a664fcdf23c756db9e6b0b53157f97105c031`.
La referencia conserva su hash anterior. Para elegirlo en el producto común:

```bash
sovengi --config configs/sovengi-memory-candidate.json chat
```

**Decisión:** integrar corpus, pesos, evaluación y configuración experimental;
mantener la referencia predeterminada. Faltan pruebas más variadas de retención,
información ausente y conversación antes de cambiar la selección general.
Ningún candidato se declara aceptado para producción.

## Diseño fijado antes de entrenar

- Padre: `reference-wiki1`, 15.769.344 parámetros; identidad y SHA-256 fijados en
  [la configuración](../../configs/sovengi-memory-stage.json).
- Nueva etapa con pesos y tokenizador propios; optimizador nuevo. No es una
  reanudación del optimizador de otra corrida, ni sustituye un entrenador activo.
- 320 ejemplos ficticios de enseñanza y 32 de validación, separados por familia
  de entidades; pares con la misma pregunta y hechos distintos. Se añade una
  muestra de al menos 160/24 registros de conversación, conservando grupos,
  mensajes, procedencia y particiones originales.
- Desarrollo: 20 consultas de memoria de entidades no entrenadas y cinco turnos
  adicionales de retención. La corrección usa la respuesta realmente generada
  como historia. Las respuestas esperadas sólo se usan al puntuar.
- Hasta 512 actualizaciones, 900 segundos de entrenamiento y dos hilos de CPU;
  el trabajo completo de Actions tiene un límite de 30 minutos. Se conservan
  checkpoints completos y se selecciona por validación, sin consultar TEST.
- Mismo `Sovengi` y `ConversationEngine` antes/después: Torch, caché habilitada,
  temperatura cero y máximo 64 tokens. La memoria usa el contrato del producto.
- Ningún mensaje personal se convierte en corpus. Los ejemplos ficticios están
  identificados como escritos por Codex; no son respuestas programadas durante
  la inferencia.

La [ejecución 38433132860](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38433132860)
publicó el candidato, corpus congelado y respuestas en `ef1e749`.
Los [checkpoints completos y registros](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38433132860/artifacts/11833021538)
están en el artefacto 11833021538, conservado hasta el 11 de enero de 2027 a las
16:45:39 UTC. Contienen estado del optimizador, pero esta etapa agotó su
presupuesto total: restaurarla no concede automáticamente más tiempo. Una nueva
etapa puede partir del candidato y debe declarar sus límites y reinicio del
optimizador. El modelo, corpus y respuestas publicados en Git no dependen del
plazo de conservación de Actions.

El recuento exacto no es la aceptación conversacional de 36/40 respuestas y
8/10 diálogos. Esta corrida ya está cerrada; no sigue entrenando en segundo plano.

## Ramas revisadas

La [auditoría de ramas](branch-audit.json) fija la comprobación anterior a esta
etapa: cero PR abiertos, 13 ramas ya contenidas en `main` y una rama histórica
`artifacts/unidad-contraste-relacional-20261012` con dos commits propios.
Sus cambios únicos son un paquete de 339 MB con pesos y ocho checkpoints del
12 de octubre. No contiene código de ejecución pendiente. Parte de su
documentación ya está idéntica en `main`; el resto tiene versiones posteriores.

Se conserva el [archivo histórico en su rama original](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012),
con SHA-256 del ZIP `5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`.
No se duplican esos 339 MB en el árbol actual ni se revierte documentación para
hacer desaparecer una rama del listado. Por tanto, existe una rama de archivo
fuera de `main`; no se afirma falsamente que todas sus revisiones estén fusionadas.

## Reproducir

Desde un checkout limpio del commit de preparación
`9dd39f55be335b924881d10445cfc78a288e483e`, con dependencias conversacionales y
Node.js instalados:

```bash
python -u examples/train_sovengi_memory.py
```

La herramienta rechaza sobrescribir el corpus congelado y verifica el padre y
el corpus de repaso antes de generar o entrenar. Por eso no debe ejecutarse
sobre el árbol que ya contiene los resultados. El workflow manual de reproducción
usa ese commit de preparación, un límite de 30 minutos para el trabajo completo
y permisos de sólo lectura del repositorio; guarda su resultado en Actions.
Publicar esta entrega no dispara otro entrenamiento.

El cierre corrige además el nombre del registro que copia el script a la carpeta
de informes: el entrenador escribe `events.jsonl`, no `metrics.jsonl`. Esa
corrección es posterior a la corrida y no cambia sus pesos, corpus ni respuestas;
los eventos originales quedaron incluidos en el artefacto completo de Actions.
