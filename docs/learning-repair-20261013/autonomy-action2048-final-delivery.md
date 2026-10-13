# Cierre del ajuste propio ACTION con contexto 2.048

La etapa terminó en **256 actualizaciones del optimizador**, tras
**187,0496777519984 segundos**, por su límite absoluto configurado. Se verificaron
la salida del trabajador y su recibo final: `status=stopped`, `reason=signal` y
`cooperativeStopReasons=[absolute_step_boundary]`.

BEST y LAST son checkpoints nativos distintos del paso 256. Comparten exactamente
los pesos, SHA256
`a2d15405f03edf74cc7fdeb05798e339205af930a6071f5640f16b6a980eb5e1`,
y el estado de Adam/RNG en `resume.pt`, SHA256
`6a705fd6c4c728f49dde18c6ec611a910146b6605a4213e0896b7673d3f1af48`.
Sus archivos `state.json` y `manifest.json` son distintos y se conservan por
separado. La NLL de validación seleccionada es **0,03811874364813169**; no acredita
competencia operativa ni aceptación conversacional. La exportación de inferencia
y demostración con respuestas y acciones reales pertenecen a otra entrega.

El [recibo principal](autonomy-action2048-final-delivery.json) registra procedencia,
estados, hashes y limitaciones. El
[recibo de verificación](autonomy-action2048-final-verification.json) conserva
las comprobaciones de archivos, enlaces y auditoría independiente de IA.
La etapa partió del padre propio ba70 con contexto
2.048, creó un optimizador y scheduler nuevos y entrenó los **15.769.344
parámetros**. Su corpus tiene TRAIN 192, VALIDATION 24 y TEST 0. Se conservaron
el tokenizador propio y la supervisión exclusiva de la última respuesta.

## Exposición TRAIN ejecutada

El [recibo exacto](autonomy-action2048-final-target-exposure.json) y el
[registro por ventana](autonomy-action2048-final-window-exposure.csv) derivan los
totales del índice y etiquetas TRAIN, orden nativo y RNG completo, reproducidos
con la semilla 20261013.

| Posiciones procesadas por BEST y LAST | Total |
| --- | ---: |
| Ventanas consumidas | 512 |
| Tokens de entrada sin padding | 519.866 |
| Etiquetas supervisadas, incluido EOS | 36.385 |
| Etiquetas EOS | 512 |
| Etiquetas de respuesta, excluido EOS | 35.873 |

Son dos épocas completas de 192 ventanas y las primeras 128 del tercer orden.
El lote lógico es dos y el microbatch máximo uno; no hay lotes finales cortos.
Cada época tiene 194.955 posiciones de entrada y 13.625 etiquetas supervisadas.
Las ventanas cacheadas tienen de 791 a 1.265 posiciones de entrada, dentro de
contexto 2.048; sus etiquetas supervisadas van de 60 a 87 y terminan con un EOS.
Estas cifras excluyen el entrenamiento heredado, el padding y la ponderación de
la pérdida. No demuestran conocimiento único ni generalización.

## Recuperar los estados posteriores al entrenamiento

[LAST](../../checkpoints/learning-repair-20261013/autonomy-action-v1/action2048-final/LAST/step-000000256-ecc22755d8084d4994ff315534a0b1d1/)
conserva **189.388.101 bytes nativos** en nueve partes de hasta 32 MiB. Se
verificaron los hashes de cada parte y de los archivos completos recorriendo las
partes en orden. Se preservan pesos, Adam/RNG, scheduler, configuración, orden y
los nombres originales BEST/LAST.

[BEST-reference](../../checkpoints/learning-repair-20261013/autonomy-action-v1/action2048-final/BEST-reference/step-000000256-df25991a5d06418fb5970aa846ae3eda/)
preserva los dos archivos propios distintos y referencia las partes idénticas
de LAST. Se comprobaron los cinco flujos completos de reconstrucción contra el
checkpoint BEST original. Su directorio de referencia no se carga directamente
como un checkpoint y evita duplicar pesos y Adam/RNG.

Desde la raíz del repositorio, con destinos nuevos:

```bash
mkdir -p private/action2048-final-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/autonomy-action-v1/action2048-final/LAST/step-000000256-ecc22755d8084d4994ff315534a0b1d1 \
  private/action2048-final-recovery/checkpoints/step-000000256-ecc22755d8084d4994ff315534a0b1d1
python3 docs/learning-repair-20261013/restore_action_native_reference.py \
  checkpoints/learning-repair-20261013/autonomy-action-v1/action2048-final/BEST-reference/step-000000256-df25991a5d06418fb5970aa846ae3eda \
  private/action2048-final-recovery/checkpoints/step-000000256-df25991a5d06418fb5970aa846ae3eda
```

Son instrucciones de recomposición de bytes, aún no ejecutadas. No importan
Torch, no interpretan pickle ni arrancan entrenamiento. Conservan los nombres
originales para resolver ambos punteros nativos. La recuperación del entrenador
también requiere el corpus con SHA256
`a7c5334e532dca161f26e6205260b4f3d5989bcbb007cfa5b2728c86bdf50396`,
el diseño y código congelados. El límite de 256 pasos ya fue alcanzado y no se
reinicia al restaurar.

## Evidencia anterior conservada y límite de recuperación

El artefacto padre ba70 sigue intacto, con pesos SHA256
`80cc46e5dfb2413355af2ce63e1c3aa6efae3760cd58c5d9f6d35b6ec270ca4a`.
Se preservaron sus metadatos, el diseño, la declaración nativa `optimizerReset`
y el evento agregado de validación del paso cero. El checkpoint nativo inicial
del ajuste ya no está presente: la retención dejó sólo BEST256 y LAST256. La
búsqueda independiente de IA tampoco encontró un estado/resume inicial
preservado bajo `/workspace` o `/tmp`.

**El optimizador/RNG nativo original anterior a los 256 pasos no es recuperable
con los archivos encontrados.** El artefacto de inferencia del padre y el evento
del paso cero no lo sustituyen. No se fabricó un estado inicial para presentarlo
como preservado.

La [prueba de creación del contexto ampliado](context2048-extension-delivery.json)
registra los tensores heredados byte por byte al crear el padre sin ajustar.
Después se entrenaron todos los parámetros. Esa prueba anterior no se interpreta
como conservación de los tensores heredados después del entrenamiento ACTION.

La preparación usó biblioteca estándar, una CPU, prioridad `nice +10`, hasta
512 MiB de espacio de direcciones y 120 segundos por operación. No cargó modelos,
Torch o pickle, no leyó corpus, cachés VALIDATION/TEST o evaluaciones reservadas,
no ejecutó pruebas, inferencia ni entrenamiento y no modificó fuentes, trabajador
o supervisor. La revisión es de IA del proyecto. La subida a GitHub se verifica
por separado.
