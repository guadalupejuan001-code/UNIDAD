# Bucle progresivo: procesos, cola y límites

Lectura del 13 de octubre de 2026, **10:53:17 UTC**. Tratamiento y control
terminaron en 4.096 updates cada uno. El tratamiento cerró en 3.337,367299398
segundos; BEST sigue en 1.024. El [control](natural-latam-v4-control-final-delivery.md)
cerró en 3.321,1108099389967 segundos, con 4.402.874 targets supervisados y
32.765 ventanas; BEST 0 es el padre heredado, sin aprendizaje nuevo del control.

La siguiente etapa de acciones llegó a **256 updates**, 187,0496777519984
segundos, todos los 15.769.344 parámetros entrenables, BEST 256 y NLL de acciones
0,03811874364813169. PID 25371, `birth=3029565`, ya está `stopped`; el
supervisor 22557 ya inició la base causal amplia v3: PID 25651,
`birth=3078774`, `step=1512`, 1.346,434966461 segundos, desde el propio
`e202eafc…`. La fuente nueva TRAIN/VALIDATION está realmente entrenándose,
SHA `81c41448…`, con configuración `ff402d5a…`. BEST causal 1.280 tiene NLL
3,285676974368285; no se compara con la NLL del corpus de acciones. La generación/evaluación de esta etapa se registra por
separado; reducir NLL no acepta trabajo autónomo. No se promovió un candidato.

El [probe regional](comparison/latam-knowledge-development-probe-v1/README.md)
obtuvo 0/12 significados TRAIN y 0/6 familias en padre, ambos checkpoints de
tratamiento y LAST control, según ambas revisiones de IA. Su primaria es del
mismo autor del probe, vínculo declarado; no existe revisión humana independiente.

La [activación del collector V2](collector-v2-enable-receipt.json) sustituyó sólo
el supervisor el 13 de octubre a las 09:09:17 UTC. Adoptó el mismo entrenador
vivo en el paso 3.333, conservando fuente, corpus, configuración, optimizador,
RNG y presupuestos. No envió señales al entrenador. La
[migración](collector-v2-migration-receipt.json) conserva el cursor 593 y copias
verificadas de raws 593–595; no reinicia el historial de ingesta.

La [configuración activa distribuida](../../examples/learning-repair-20261013/configs/continual-live-collector-v2.json)
tiene SHA `24c561cc5a71132deb9321bdf356e46100a1820d36b62f6a706af3f26588ccf4`.
El collector está habilitado para las siguientes esperas sin entrenamiento
activo. La [reparación](collector-v2-design.md) separa cinco clases de párrafos
históricos repetidos de las colisiones nuevas. Su preview preparó 109 artículos
nuevos y conservó 2.299 anclas; sigue `prepared_not_admitted`, sin entrenamiento
ni admisión de ese preview.

La cola conserva este orden y límites propios:

| Etapa posterior | Estado en esta lectura | Límite |
| --- | --- | --- |
| Control natural | Terminado 4.096; BEST 0 heredado | 4.096 updates y umbral nativo de 7.200 s |
| [SFT de acciones](autonomy-action-v1-enqueue.json) | Terminado 256; demo propia fallida, mejora parcial de formato | 256 updates / 600 s; lote 2, microbatch 1 |
| [Español + LATAM causal v3](wide-latam-v3-enqueue.json) | Primer shard activo; segundo en cola | Cada uno: 2 épocas / 2.048 updates / 1.800 s |
| Retención natural + acciones | Encolada tras los dos shards, sin entrenar | 1.024 updates / 1.800 s |
| [Copia de argumentos V2/V1](autonomy-action-copy-v2-enqueue.json) | Aprobado y en cola tras v3/retención, sin entrenar; mismo padre `e202…` | 256 updates / 600 s por brazo |
| [Tatoeba posterior](tatoeba-next-v1-enqueue.json) | Causal y retención encolados después, sin entrenar | Causal: 2 épocas / 2.048 / 1.800 s; SFT: 1.024 / 1.800 s |

Las etapas nuevas conservan 6 GiB de RSS y 2 GiB de disco por corrida. La mezcla
de retención tiene 10.551 TRAIN/1.783 VALIDATION, SHA
`808e1bdd98f1447fea82f83e3d936e8724f86dc23c124d8b8e3cc9504937b7a7`;
conserva las filas previas y añade acciones, sin contar lo repetido como
conocimiento nuevo. La adquisición conserva procedencia y particiones.

`maxCycles=null` permite proseguir entre etapas mientras proceso, entorno,
datos y recursos lo permitan; cada etapa conserva su límite finito. No
garantiza ejecución fuera del entorno, mejora automática, aceptación de pesos
ni reapertura de los presupuestos históricos 24+2 horas. Para leer el estado:

```sh
python3 examples/continual_training.py status \
  --config examples/learning-repair-20261013/configs/continual-live-collector-v2.json
```

La entrega está integrada mediante [PR10 fusionado sin borrador](https://github.com/guadalupejuan001-code/UNIDAD/pull/10),
main `62cf4d43780204f8d09dce6715cefcb6110aa310`, con los cuatro checks de
`38379096070` y `38379113739` aprobados. El
[recibo de publicación](pr10-publication-verification.json) verifica código,
configuración, datos y pesos; el [recibo de correcciones](pr10-ci-corrections.json)
conserva el primer fallo sin tocar el entrenamiento nativo. La nueva generación
LAST 4.096 se prepara por separado con dos revisiones terminadas: ambas dan
0 completas, una parcial, 39 fallidas, 14 críticas y una técnica; no hay
aceptación ni mejora conversacional verificada. Integrar artefactos no acepta la conversación. El artefacto extendido inicial
permanece como baseline histórico; la etapa nueva de acciones sí ejecutó 256
updates, sin aceptación de capacidad aprendida todavía.

La [prueba real de acciones256](autonomy-demonstration/action256-own-v1/README.md)
produjo una propuesta JSON válida con `k.txt`, rechazada antes del handler porque
sólo estaban autorizados `dos.txt`/`uno.txt`; el nombre no se corrigió. La tarea
mínima siguió dando JSON inválido. El [reintento del host](operational-autonomy.md)
permite nuevas propuestas tras rechazos de argumentos de lectura, por defecto
cero, con fallo persistido y permisos iguales; la [prueba real de reintento](autonomy-demonstration/action256-own-read-repair-v1/README.md)
también terminó fallida: segundo JSON mal formado, sin efectos ni informe. Ningún resultado promueve al trabajador automáticamente.
