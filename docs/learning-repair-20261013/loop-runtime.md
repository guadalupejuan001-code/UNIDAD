# Bucle progresivo: procesos, cola y límites

Lectura del 13 de octubre de 2026, **09:11:51 UTC**. El tratamiento natural +
LATAM sigue activo: PID 19896, `birth=2318347`, `step=3560`,
`elapsedSeconds=2880.2897863769977`. Su mejor validación hasta esta lectura es
el paso 1.024, NLL 1,7603046832825393; no se compara directamente con NLL de otro
corpus. El supervisor es PID 22557, `birth=2592201`. El brazo control sigue
pendiente y ningún candidato se ha promovido.

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
| Control natural | Pendiente, mismo padre que tratamiento | 4.096 updates y umbral nativo de 7.200 s |
| [SFT de acciones](autonomy-action-v1-enqueue.json) | Encolado tras ambos brazos, sin entrenar | 256 updates / 600 s; lote 2, microbatch 1 |
| [Español + LATAM causal v3](wide-latam-v3-enqueue.json) | Dos shards encolados tras acciones, sin entrenar | Cada uno: 2 épocas / 2.048 updates / 1.800 s |
| Retención natural + acciones | Encolada tras los dos shards, sin entrenar | 1.024 updates / 1.800 s |
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
