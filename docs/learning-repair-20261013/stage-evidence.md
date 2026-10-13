# Evidencia de la nueva etapa lingüística

Este documento reúne la contabilidad y la procedencia de
`private/learning-repair-20261013/runs/foundation-15m`, incluidas observaciones
durante su ejecución y su cierre por tiempo. Es trabajo de una IA del proyecto.
La etapa causal terminó en step 8.914; no contiene una aceptación conversacional
ni sustituye la evaluación posterior del ajuste SFT.

## Identificación del diseño leído

El diseño registrado a las **2026-10-13 04:11:28 UTC** identifica una base causal
iniciada desde pesos aleatorios, BPE de 2.048 tokens, contexto de 512 y 15.179.520
parámetros entrenables. La arquitectura declarada tiene ocho capas, dimensión 384,
ocho cabezas y multiplicador feedforward cuatro. El entrenamiento lógico usa
batch ocho, microbatch cuatro, cuatro hilos PyTorch y BF16 en CPU.

La tasa base es 0,0006, con warmup de 128 updates y coseno hasta el update 12.000,
donde queda en el 10 % de la tasa base. El límite solicitado es 12.000 updates
y 7.200 segundos; una lectura del diseño no demuestra que se hayan completado.

El diseño vincula el corpus train/validation de 27.132.589 bytes con SHA256
`5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17`.
La identidad de caché vincula el tokenizador con SHA256
`848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720`.
Esos identificadores proceden del diseño y del manifiesto de caché leídos.
Root confirmó que código, configuración y corpus estaban congelados y autorizó
las copias exactas de los tres manifiestos. El
[recibo de entradas](foundation-15m-inputs-receipt.json) conserva sus tamaños y
hashes; el [diseño](foundation-15m-stage-design.json) incluye los hashes de código
del lanzamiento, y los manifiestos de [caché](foundation-15m-cache-manifest.json)
y [corpus](foundation-15m-corpus-manifest.json) conservan su procedencia.

## Contabilidad de ventanas sin cargar el modelo

| Partición en la caché | Ventanas | Posiciones objetivo por pasada | Media por ventana | Ventanas completas de 512 |
| --- | ---: | ---: | ---: | ---: |
| Train | 17.995 | 5.780.408 | 321,223006 | 7.281 |
| Validation | 2.189 | 687.867 | 314,238008 | 839 |
| TEST | 0 | 0 | — | 0 |

Se leyeron solamente los índices `<QQ>` de la caché: offset en bytes y longitud
de cada ventana. La lectura sumó **322.944 bytes**. Sus hashes coinciden con el
manifiesto, y en cada partición se cumple
`bytes de ventanas = posiciones × 8`, porque cada posición guarda un input y
una etiqueta `int32`. La política congelada es `causal_lm` con ventanas
`contiguous_nonoverlapping`: cada posición supervisa la predicción del siguiente
token y no hay etiquetas ignoradas por solapamiento.

Este total cuenta posiciones del corpus codificado en una pasada. No cuenta
valores distintos del vocabulario ni certifica conocimientos distintos. Repetir
épocas aumenta posiciones procesadas y no aumenta el texto del corpus. El
[recibo de contabilidad](foundation-15m-token-accounting.json) conserva también
el histograma completo de longitudes y la identidad de la caché.

Si se alcanzaran los 12.000 updates, habría 2.250 updates por época, cinco épocas
completas y 750 updates adicionales. `12.000 × 8` da 96.000 draws nominales;
las cinco últimas tandas de época tienen tres ventanas cada una, de modo que
el diseño realmente procesaría **95.975 ventanas**. Eso equivale a unas
**5,3334 pasadas**. Aplicar la media a todos los draws nominales estima 30,8374
millones de posiciones; contar las tandas cortas estima **30,8294 millones**.
La parte de cinco épocas completas sí cuenta 28.902.040 posiciones exactas;
la parte barajada incompleta requiere el `order/cursor` final para ser exacta.
Estas cifras describen exposición prevista si se alcanza el límite, no trabajo
ya completado ni una previsión de horas.

Los eventos nativos de entrenamiento registran step, NLL y learningRate, pero
no el número de objetivos del batch. Durante la corrida, `step × batch × media`
sería sólo una estimación basada en la media. El total procesado exacto se
reconstruirá al cierre desde el estado nativo completado: épocas completas más
las longitudes de los índices incluidos en `order[:cursor]`. Un checkpoint final
permite contabilizar ese prefijo sin cargar pesos ni estado Adam.

## Observación de recursos

A las **2026-10-13 04:12:56 UTC**, el PID **9540** existía y su nacimiento
`805298` coincidía con `status.json`. `/proc/9540/status` registraba estado
`R (running)`, RSS de **582.704 KiB**, máximo RSS de **603.736 KiB** y 16 hilos
del proceso. Los cuatro hilos configurados en PyTorch y el total de hilos del
proceso son medidas diferentes.

La CPU acumulada del proceso era **291,49 segundos** según 28.430 ticks de
usuario y 719 de sistema, con 100 ticks por segundo. Esa cifra acumula ejecución
de varios hilos; no equivale al tiempo de pared del entrenamiento.

La lectura del filesystem registraba **26.699.030.528 bytes disponibles** en
`/workspace/UNIDAD`. Es una observación del filesystem completo y no un límite
de espacio asignado a la corrida. Los techos del diseño son 6.144 MiB de RSS
y 2.048 MiB para el directorio de la corrida, sujetos a parada cooperativa.

En esa observación, el estado nativo seguía en step 0 y no había checkpoint ni
NLL de validación registrados. El PID y sus ticks acreditaban un proceso activo;
no se atribuyen updates a partir del consumo de CPU.

Una lectura posterior a las **2026-10-13 04:16:46 UTC** registró `status=running`,
step 260 y 315,169 segundos acumulados del estado nativo. El último evento de
entrenamiento completo leído ya estaba en step **264**, con NLL de batch
**5,1676707** y learningRate **0,00059982517**. La validación inicial había quedado
registrada en **7,7110481**. Las dos pérdidas corresponden a muestras distintas;
su diferencia no acredita una mejora validada. Había un checkpoint de step 0,
creado antes de los updates, que no se copió para esta documentación.

## Recibos que completan la entrega

| Evidencia | Fuente prevista | Regla de incorporación |
| --- | --- | --- |
| Diseño, configuración y hashes de código congelado | `runs/foundation-15m/stage-design.json` | Copia exacta incorporada tras confirmación de root. |
| Identidad, índices y hashes de ventanas | `runs/foundation-15m/tokenized-cache/manifest.json` | Manifiesto final confirmado incorporado; ventanas binarias no copiadas. |
| Procedencia, revisión, licencias y particiones de datos | `corpus/causal-spanish-v1/manifest.json` | Manifiesto congelado incorporado tras confirmación de root. |
| Totales realmente procesados y estado de parada | Estado del checkpoint final y recibo del runner | Leer sólo después del cierre; conservar la distinción entre máximo configurado y completado. |
| Pesos, tokenizador y publicación | Entrega de root | Identificar hashes y destino confirmados al cierre. |
| Capacidad conversacional | Evaluación específica autorizada | Mantener separada de pérdida causal y datos de infraestructura; aquí no se generan respuestas. |

Las observaciones anteriores no copiaron estados dinámicos ni pesos. La
siguiente incorporación conserva un checkpoint cerrado, publicado por el
entrenador mediante rename; no copia su directorio temporal ni un estado en
escritura.

## Respaldo cerrado de step 3.072

Se capturó `step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86` en staging
independiente. Sus cinco hashes originales coincidieron antes de copiar,
en la copia y después de copiar. Se empaquetó esa copia en nueve partes de
como máximo 32 MiB, conservando pesos, Adam/scheduler/RNG, tokenizador y estado.
El [recibo de entrega](foundation-15m-step-3072-delivery.json) identifica
182.406.100 bytes nativos y SHA256 del manifiesto del paquete
`7c6534c70916993e291ab25fdc6a06806294f054c4b2d9dc5ae3b8db95ac2b86`.

El `state.json` congelado tiene SHA256
`34d8e61b03677fd1cc177ac4f19cdca893bb3b508f98d4f231671d67b4c553e3`.
Registra 3.072 updates completos, 2.597,678 segundos acumulados, una época
completa y cursor 6.576 de la siguiente. La NLL de validación allí guardada
es 3,7733837676747695. No se atribuyen estos datos al cierre de la corrida.

Se reconstruyó la exposición exacta desde ese estado y el índice train
verificado: una pasada aporta 5.780.408 posiciones y el prefijo
`order[:6576]` aporta 2.106.332. El total es **7.886.740 posiciones objetivo**
en **24.571 ventanas**, cinco ventanas menos que `3072 × 8` por la última
tanda corta de la primera época. El
[recibo de exposición](foundation-15m-step-3072-target-exposure.json) conserva
la fórmula y los hashes; no se cargaron pesos, Adam, etiquetas, corpus ni
casos de evaluación para ese cálculo.

El [procedimiento de transporte y recuperación](checkpoint-packaging.md)
distingue reponer bytes y reanudar el entrenamiento. Este respaldo intermedio
no promueve un candidato ni declara conversación competente. Las operaciones
de entrega no ejecutaron pruebas de software, restauración ni entrenamiento.

## Cierre de la base causal

La lectura cerrada a las **2026-10-13 06:13:54 UTC** registró
`completed`, motivo `total_time_budget`, step **8.914** y **7.337,687718068
segundos** acumulados. El worker PID 11176, nacimiento 1165743, ya estaba muerto;
`stage-latest.json` correspondía al mismo cierre y confirmaba código y entradas
sin cambios. El máximo configurado de 12.000 updates no se completó. El umbral
de tiempo de 7.200 segundos provoca una parada cooperativa; validación final y
guardado explican que el contador cerrado lo exceda.

El último recibo del runner mide máximo RSS de **3.306.389.504 bytes** y
máximo observado de **661.009.654 bytes** en el directorio nativo. Son medidas
de esa invocación, incluidas exportaciones, distintas de los techos configurados
de 6.144 MiB y 2.048 MiB. No describen la RAM ni el disco de todos los servicios
del entorno.

Se capturaron BEST y LAST después del cierre, con cinco archivos originales
cada uno, y se copiaron los cuatro archivos de la exportación final. Los
checkpoints tienen UUID distintos en el mismo step; se conserva cada uno.
Cada paquete nativo contiene **182.406.109 bytes** en nueve partes. El artefacto
de inferencia final contiene **60.784.468 bytes** e identifica
`unidad-conversation-ee24651fb8cc3a01e1acbb57db5f40222a6b9b5fe150dbff7186b961945fa398`.
El [recibo final de entrega](foundation-15m-final-delivery.json) conserva las
catorce identidades de archivo original, verificaciones antes/copia/después,
partes, rutas y hashes. Se añadieron 790.414.254 bytes entre staging, paquetes
y exportación; la entrega usó buffers de 1 MiB, un hilo y prioridad nice 10.

El [recibo de exposición final](foundation-15m-final-target-exposure.json)
reconstruye exactamente tres épocas completas, con **17.341.224 posiciones**,
y el prefijo de 17.312 ventanas de la siguiente, con **5.565.445 posiciones**.
El total es **22.906.669 posiciones objetivo** y **71.297 ventanas** para BEST
y LAST. No se interpretaron etiquetas, pesos ni estado Adam para contarlas.

La selección causal final registra NLL **2,981873986010036**. Esa pérdida
compara el mismo conjunto de validación con la referencia inicial de 7,7110481;
acredita una reducción de pérdida causal, no conversación competente.
El modelo sigue experimental. La primera exportación de step 4.096 permanece
como historia y el ajuste SFT posterior se identifica como otra etapa, con su
propio optimizador y corpus. No se ejecutaron restauraciones, pruebas de
software ni generación de conversaciones al capturar esta entrega.
