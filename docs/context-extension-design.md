# Extensión de contexto propia: preparación sin entrenamiento

El proponente real de acciones encontró un bloqueo de longitud: el marco con
tres herramientas ocupa 732 tokens frente al contexto de 512 del modelo propio.
Un marco reducido a una herramienta cabe en 336 tokens, pero el modelo
`unidad-conversation-072eedd276160f7581c5a70b57e66ae3000d2e664ed32f6b5a4669d4d12dcf57`
produjo texto libre de rechazo, no la acción JSON requerida. La capacidad de
alojar el marco completo y el aprendizaje del formato son problemas pendientes
que requieren evidencia por separado.

La medición posterior de las secuencias reales de ajuste de acciones, con las
tres herramientas y el objetivo supervisado, da 816, 1064 y 1298 tokens. El
contexto de 1024 todavía recorta instrucciones o hechos necesarios en las
secuencias largas; el tercer caso también excede ese límite. Por eso se admite
una extensión explícita a 2048. Esta medida corrige la capacidad de alojar esas
secuencias; no demuestra que el modelo ya haya aprendido a generar acciones.

[extend_own_conversation_context.py](../examples/extend_own_conversation_context.py)
prepara un artefacto nativo con contexto de 1024 por defecto o de 2048 mediante
una opción explícita, usando el mismo Transformer,
tokenizador y pesos propios. El código genérico ya admite esa configuración; no
se modifica `src/ia_propia/conversation` ni se migra a otra arquitectura.
El artefacto nuevo se identifica como **extensión sin entrenar y sin promover**.
Esta entrega sólo incluye el procedimiento y pruebas con modelos sintéticos
pequeños; no publica una extensión completa del modelo de 15 millones.

El procedimiento exige una configuración completa elegida por el anfitrión.
`load_artifact(..., expected_model_config=...)` verifica esa configuración y los
hashes antes de admitir los pesos. Se exige procedencia propia válida desde
`random_weights`, incluidos identidad del padre, SHA de pesos y SHA del corpus.
La exportación contiene `trainedFrom: own_checkpoint`, linaje del padre
inmediato y un bloque `contextExtension` que conserva también su procedencia.
El SHA del corpus identifica únicamente el corpus heredado: no se abre ni se
entrena con él en esta operación. No se copian métricas, pasos o perfiles del
entrenamiento anterior como si pertenecieran a la extensión.

Se construye un esqueleto nativo en `meta`, se reutiliza el snapshot CPU
verificado y se asigna una matriz posicional nueva. Las filas `[0, 512)` se
copian sin cambiar ningún bit; todas las demás matrices se conservan completas.
Sólo las filas `[512, contexto_nuevo)` se inicializan con una normal de media cero y
desviación 0,02 usando un generador CPU local y la semilla propia registrada
(o una semilla propia explícita). No se consume el generador aleatorio del
llamador. Los pesos exportados siguen siendo finitos y `float32`, y el SHA del
tokenizador debe permanecer idéntico. Las pruebas verifican igualdad exacta de
logits en inferencia para secuencias que cabían en el contexto anterior, además
de un paso de inferencia finito que usa posiciones nuevas. Eso no acredita
conversación ni planificación aprendida con esas posiciones.

La salida siempre es una ruta nueva. Sin `--output`, se crea un nombre único
junto al padre. Se rechazan destinos existentes y rutas dentro del artefacto
original; la reserva exclusiva del directorio evita reemplazarlo. La operación
usa un hilo Torch CPU y admite como máximo 2048 posiciones y 20.000.000 de
parámetros, aunque el contrato genérico acepte arquitecturas mayores. Tiene
comprobaciones de plazo entre operaciones; `timeout` fija además el límite
externo del proceso completo. Un fallo de escritura puede dejar una exportación
incompleta en la ruta nueva; el cargador nativo la rechaza sin un manifiesto
final válido.

Para el padre que se seleccione después de la corrida activa:

```bash
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/extend_own_conversation_context.py /ruta/al/artefacto-propio \
  --expected-config /ruta/a/configuracion-propia-seleccionada.json \
  --context-length 1024 --output /ruta/nueva/extension-sin-entrenar
```

Para preparar el contexto elegido para las secuencias de acciones, se cambia
únicamente la opción a `--context-length 2048`, usando siempre una ruta nueva.
El padre seleccionado después de 1024 actualizaciones SFT sigue siendo un
modelo propio con contexto de 512; el número de actualizaciones no es su
longitud de contexto. Se conservan todos sus pesos aprendidos. Las posiciones
añadidas siguen sin entrenar hasta que se ejecute una etapa posterior.

La configuración seleccionada se fija antes de ejecutar; el cargador admite
únicamente esa arquitectura y registra en el linaje los hashes del snapshot
realmente cargado. El nuevo directorio es una
entrada válida para `training._initial_own_model` con su configuración de 1024
o de 2048.
La prueba técnica comprueba esa admisión sin leer corpus ni TEST. Una etapa SFT
posterior debe usar un optimizador nuevo y datos con procedencia y particiones
congeladas; la operación no reanuda el optimizador anterior. La cabeza opcional
de copia, si existe, conserva sus pesos y sus límites nativos de supervisión
auxiliar de 512; ampliar la posición no cambia esas reglas.

## Coste previsto, sin medición de entrenamiento

Para `d_model=384`, ampliar 512 posiciones añade exactamente
`512 × 384 = 196.608` parámetros. El modelo de 15.179.520 pasaría a 15.376.128;
la carga FP32 adicional de pesos sería de 786.432 bytes (0,75 MiB), excluyendo
cabeceras. Pesos, gradientes y dos momentos Adam FP32 sumarían aproximadamente
235 MiB sólo para esos cuatro grupos de tensores del modelo extendido. Esa
cuenta no incluye activaciones, intermediarios, ventanas, exportaciones o el
proceso de Python y no constituye un presupuesto de RAM medido.

Procesar una ventana completa de 1024 en vez de 512 duplica las partes lineales
por token y cuadruplica el tamaño teórico de las matrices de atención. El coste
real depende de secuencias, lote, implementación y proyección supervisada;
no se ha medido para esta extensión. Ampliar de 512 a 2048 añade
`1536 × 384 = 589.824` parámetros: el modelo resultante tendría 15.769.344 y
2.359.296 bytes (2,25 MiB) adicionales de pesos FP32. A longitud completa de
2048, las partes lineales por token serían cuatro veces las de 512 y las
matrices teóricas de atención dieciséis veces mayores. La RAM y la latencia
reales siguen sin medir; estas proporciones no predicen el tiempo total.
El BF16 opcional del entrenador afecta
parte del cálculo y las activaciones, mientras parámetros, pérdidas y estados
Adam siguen siendo FP32. No se atribuye ahorro ni tiempo real de SFT aquí.

La siguiente prueba de aprendizaje de acciones prevista usa microbatch 1,
lote lógico 2, como máximo 256 actualizaciones y presupuesto de 600 segundos.
Son límites de un experimento futuro, no una corrida ejecutada por este helper.
La conversación y las acciones realmente generadas deberán evaluarse aparte;
el artefacto extendido no se promueve al exportarlo.

Validación técnica acotada:

```bash
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_own_context_extension.py -v
```

[Las pruebas](../tests/test_own_context_extension.py) usan sólo artefactos
sintéticos creados en directorios temporales. Las respuestas reservadas, los
datos personales y el artefacto propio de la corrida activa no se cargan.
