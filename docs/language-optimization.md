# Evaluación del lenguaje con menos memoria

La evaluación de pérdida conserva ahora sólo el último estado recurrente. Antes
guardaba todos los estados del fragmento aunque no calculase gradientes. El
entrenamiento conserva el historial necesario para BPTT y usa el mismo recorrido
aritmético. No se reentrenó ni modificó el modelo guardado.

En una medición local con 256 caracteres, 128 unidades ocultas y vocabulario de
47 símbolos, el pico de asignaciones rastreadas bajó de **298.976 a 6.040 bytes**,
aproximadamente un **98 %**. Son asignaciones temporales de una llamada medidas
con `tracemalloc`; no representan la memoria total del proceso ni incluyen los
pesos ya creados. Se repitió la medición siete veces por versión, con las mismas
entradas y parámetros.

La mediana de tiempo sin rastreo fue 4,015 ms antes y 3,961 ms después en 21
repeticiones por versión. Esa diferencia pequeña no demuestra una aceleración
general. El beneficio verificado es la reducción de memoria durante evaluación.

Las pérdidas y todos los gradientes BPTT coincidieron exactamente con el código
anterior. Las métricas de los tres bloques del modelo guardado y su generación
con la misma seed también quedaron exactamente iguales; su pérdida de prueba
sigue siendo 2,201882 nats por carácter. Los hashes del archivo y del modelo se
conservaron. Las diez pruebas del lenguaje aprobaron, incluida la comprobación
numérica de gradientes.

Los valores, repeticiones, hashes de código, commit anterior, parámetros de
medición y comprobaciones se guardan en
[language-optimization.json](language-optimization.json). Esta optimización no
cambia la calidad lingüística ni acredita nuevas capacidades del modelo.
