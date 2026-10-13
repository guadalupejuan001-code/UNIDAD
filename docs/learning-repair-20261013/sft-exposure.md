# Exposición supervisada real del ajuste conversacional

El recibo [del paso 2048](sft-exposure-step002048.json) conserva el estado de
contabilidad de un checkpoint inmutable del ajuste `conversation-15m`. En ese
momento se habían presentado **1.323.952 posiciones supervisadas** al cálculo
de pérdida, mediante 16.384 ventanas de entrenamiento. Se incluyen los EOS
supervisados; no se cuentan como objetivos los mensajes anteriores del contexto.

La pasada completa contiene 18.588 ventanas, **1.500.021 objetivos** y
4.449.455 posiciones de entrada sin padding. Contar la longitud total de las
entradas como objetivos de SFT habría sobreestimado esta exposición: en el
checkpoint contado había 3.920.788 posiciones de entrada y 1.323.952 objetivos.
La reducción `sequence_mean` normaliza cada ventana antes de combinar la
pérdida. Por ello, estos recuentos tampoco representan porcentajes de gradiente
aportados por cada fuente.

El [utilitario estándar](../../examples/audit_learning_exposure.py) comprueba
hashes y lee únicamente las etiquetas de las ventanas **TRAIN** del mmap,
junto con `state.json`. No carga pesos, pickle de Adam, Torch, tokenizador,
ventanas de validación o TEST, ni evaluaciones. Conserva la permutación de
entrenamiento usada para reproducir el recuento. Sólo admite muestreo uniforme
sin reemplazo; rechaza las políticas adaptativas o ponderadas cuya exposición
requiera una contabilidad diferente.

```bash
nice -n 19 python examples/audit_learning_exposure.py \
  --cache private/learning-repair-20261013/runs/conversation-15m/tokenized-cache \
  --state RUTA_AL_CHECKPOINT_INMUTABLE/state.json \
  --output RUTA_NUEVA_AL_RECIBO.json \
  --max-seconds 60
```

El proceso se limita a una CPU y 512 MiB, comprueba la identidad de todos sus
archivos antes y después y conserva cada recibo sin sobrescribir evidencia
distinta. El cálculo ejecutado para este recibo tardó 0,502 segundos según el
ejecutor. No se detuvo, reinició ni alteró el trabajador de entrenamiento.

Este dato acredita exposición al entrenamiento, no adquisición de conocimiento,
generalización ni conversación competente. El recibo es una auditoría de IA del
proyecto, sin revisión humana independiente. Los pesos finales y las respuestas
generadas se entregan y evalúan por separado después del cierre de la etapa.

## Selección intermedia de la primera invocación

El [recibo del paso 4096](sft-exposure-step004096.json), preparado después de la
primera pausa y con la corrida ya reanudada, cuenta **2.638.841 objetivos** y
32.764 ventanas: una época completa y 14.176 ventanas de la siguiente. Las
entradas sin padding suman 7.841.381 posiciones y se conservan separadas de los
objetivos. Sus hashes de TRAIN y del checkpoint no cambiaron durante la
auditoría. Este es el checkpoint BEST seleccionado para la primera exportación
intermedia, mientras LAST de la pausa quedó en el paso 5.073 para reanudar.
La auditoría de este recibo tardó 5,237 segundos según el ejecutor, sin
interrumpir el entrenamiento ni cargar el modelo.
