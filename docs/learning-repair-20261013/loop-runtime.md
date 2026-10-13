# Bucle progresivo: procesos, cola y límites

Lectura del 13 de octubre de 2026, **11:28:38 UTC**. Tratamiento y control
terminaron en 4.096 updates cada uno. El tratamiento cerró en 3.337,367299398
segundos; BEST sigue en 1.024. El [control](natural-latam-v4-control-final-delivery.md)
cerró en 3.321,1108099389967 segundos, con 4.402.874 targets supervisados y
32.765 ventanas; BEST 0 es el padre heredado, sin aprendizaje nuevo del control.

La siguiente etapa de acciones llegó a **256 updates**, 187,0496777519984
segundos, todos los 15.769.344 parámetros entrenables, BEST 256 y NLL de acciones
0,03811874364813169. PID 25371, `birth=3029565`, ya está `stopped`; el
supervisor 22557 completó los dos shards causales v3 y retención. [LANG1](wide-latam-v3-language-1-final-delivery.md),
PID 25651/`birth=3078774`, cerró 2.048 updates, 1.931,1925520530021 s,
BEST 2.048 NLL 3,2358626570786604; fuente `81c41448…`, configuración `ff402d5a…`.
El umbral de 1.800 s era cooperativo: excedió 131,19 s; recibos conservan
`signal` nativo y `absolute_step_boundary` del runner. El artefacto propio
de [inferencia](wide-latam-v3-language-1-inference-export.json), `a8c25b01…`,
está exportado, sin aceptación de conversación o trabajo.
LANG2 cerró 2.048 en 486,66538323400164 s; retención, 1.024 en
638,6049642170037 s. Las pérdidas sobre corpus/objetivos diferentes no se comparan.

V2 tratamiento cerró 256 updates, 157,4759767220021 s, PID 27145/`birth=3403056`,
BEST 256 NLL 0,32269909729560214, modelo propio `e70ea28e…`.
El supervisor registra `generating_development`, entrenamiento inactivo y
generación viva PID 27186/`birth=3420096`; su control V1 aún no tiene estado
nativo en este snapshot. Reducir NLL no acepta trabajo autónomo. No se promovió un candidato.

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
| [Español + LATAM causal v3](wide-latam-v3-enqueue.json) | Dos shards cerrados en 2.048 cada uno; LANG1 exportado | Cada uno: 2 épocas / 2.048 updates / umbral cooperativo 1.800 s |
| Retención natural + acciones | Terminada 1.024, después de ambos shards | 1.024 updates / umbral cooperativo 1.800 s |
| [Copia de argumentos V2/V1](autonomy-action-copy-v2-enqueue.json) | Tratamiento cerrado 256, generando DEV; control pendiente; mismo padre `e202…` | 256 updates / 600 s por brazo |
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

[PR11 abierto sin borrador](https://github.com/guadalupejuan001-code/UNIDAD/pull/11)
publicó head `68b6af16d1056c2ef02d013144cf3501b81c9ff3`, con cuatro checks
SUCCESS de `38387474574`/`38387694065`. La segunda entrega con LANG1 y
serialización de JSON está local, pendiente de publicar/CI; main sigue en PR10.
Ninguna prueba técnica acepta conversación ni planificación.

La [prueba real de acciones256](autonomy-demonstration/action256-own-v1/README.md)
produjo una propuesta JSON válida con `k.txt`, rechazada antes del handler porque
sólo estaban autorizados `dos.txt`/`uno.txt`; el nombre no se corrigió. La tarea
mínima siguió dando JSON inválido. El [reintento del host](operational-autonomy.md)
permite nuevas propuestas tras rechazos de argumentos de lectura, por defecto
cero, con fallo persistido y permisos iguales; la [prueba real de reintento](autonomy-demonstration/action256-own-read-repair-v1/README.md)
también terminó fallida: segundo JSON mal formado, sin efectos ni informe. Ningún resultado promueve al trabajador automáticamente.

La corrección del adaptador serializa marcadores literales de rol como datos
JSON, escapando `<`/`>` sin omitir contenido; decodificar devuelve los valores
exactos. Sus tres regresiones llevan la verificación pertinente a 87 pruebas.
La comparación V2/V1 requiere los [intentos y efectos reales](operational-autonomy.md),
con la misma política y tareas, antes de atribuir mejora a los datos nuevos.
La [demostración propia V2](autonomy-demonstration/action-copy-v2-treatment-own-v1/evidence.json)
ya realizó una lectura autorizada de `dos.txt`; se bloqueó en la segunda
propuesta por JSON inválido, sin informe y con originales intactos.
Se conserva como mejora parcial de copia en un caso DEV expuesto, no aceptación.

Registro posterior del13 de octubre de 2026, **11:41:18 UTC**: [V2 y su control](autonomy-action-copy-v2-real-matched-development.json)
cerraron256 updates cada uno. Sólo el tratamiento ejecutó una lectura propia
autorizada; ambos fallaron el trabajo sin informe. El supervisor 22557 inició
`language-z-tatoeba-next-v1`, etapa 0 causal, PID 28222, `running`, paso 0 y 3,6537 s.
No se cuentan validación/inicialización como updates nuevos. La cola mantiene
la retención posterior de Tatoeba, sus límites y padre fijados; los snapshots
anteriores conservan su estado histórico. La [evaluación natural posterior](comparison/development-wide-latam-retention1024-v1/README.md)
dio 1/40 completa,0/10 diálogos, seis críticas/dos técnicas. No se promovió un candidato.
