# Auditoría nativa de TRAIN de natural-latam-sft-v4

La [medición completa](receipt.json) comprobó las 10.359 filas TRAIN de la rama
de tratamiento antes de su admisión: 9.951 originales y 408 regionales nuevas.
Cada fila coincide byte por byte con su componente congelado. El helper llamó
la función real `training._record_windows` con el tokenizador propio BPE de
2.048 tokens, contexto de 512, `sliding_context` y objetivo
`assistant_last_reply`. No cargó Torch ni pesos.

| Medición en tokens | Todo TRAIN | TRAIN original | TRAIN regional nuevo |
| --- | ---: | ---: | ---: |
| Filas / ventanas nativas | 10.359 | 9.951 | 408 |
| Entrada completa: mediana / p95 / máximo | 243 / 479 / 512 | 249 / 480 / 512 | 147 / 207 / 305 |
| Prompt con rol assistant: mediana / p95 / máximo | 99 / 283 / 496 | 98 / 285 / 496 | 112 / 151 / 275 |
| Texto de última respuesta: mediana / p95 / máximo | 89 / 368 / 443 | 96 / 370 / 443 | 36 / 75 / 156 |
| Targets, incluyendo EOS | 1.353.109 | 1.336.577 | 16.532 |
| Prefijo o historia perdidos en ventanas | 0 | 0 | 0 |

Todas las entradas caben completas en una sola ventana. La longitud de entrada
excluye el último EOS desplazado como etiqueta; una entrada de 512 puede tener
513 tokens en la secuencia completa. Cada target de la última respuesta y su
EOS aparecen exactamente una vez. Ningún rol, texto de respuestas anteriores
ni EOS histórico se convierte en etiqueta positiva. Las 79 filas TRAIN con
contexto assistant deliberadamente erróneo cumplen esa misma comprobación;
las diez restantes pertenecen a VALIDATION y sus cuerpos no se decodificaron.

Las filas nuevas son el 3,94 % de las ventanas TRAIN y el 1,22 % de sus tokens
objetivo. Esta diferencia procede de respuestas regionales más breves. El
contexto causal efectivo por token supervisado incluye el prefijo y los tokens
ya emitidos de la respuesta; su mediana es 210 en todo TRAIN y 133 en el lote
regional. La medición no prueba exactitud semántica, aprendizaje de pesos ni
aceptación conversacional.

La ejecución tardó 9,77 segundos y usó 109,24 MiB de RSS máximo, dentro de
1 CPU, 120 segundos y 1 GiB de espacio de direcciones. Los hashes de corpus,
tokenizador y fuentes quedaron iguales antes y después. Se omitieron cuerpos
VALIDATION/TEST antes de decodificar JSON y no se abrió ningún archivo de
evaluación reservado.

Reproducción desde la raíz del repositorio, con un destino de salida nuevo:

```bash
python docs/learning-repair-20261013/natural-latam-v4-train-audit/audit_native_windows.py \
  --repo "$PWD" \
  --package "$PWD/examples/learning-repair-20261013/natural-latam-sft-v4" \
  --tokenizer "$PWD/models/learning-repair-20261013/conversation-15m/intermediate-step004096/tokenizer.json" \
  --output /tmp/unidad-natural-latam-v4-native-train-audit.json
```

El [helper reproducible](audit_native_windows.py) rechaza cambios de identidad
en los tres archivos del paquete y en el tokenizador, comprueba estabilidad de
las fuentes y se niega a reemplazar un recibo existente. La salida conserva
distribuciones de todo TRAIN, TRAIN original y TRAIN regional, conteos por
fuente y conducta, hashes y límites. No publica texto de mensajes.

La [revisión mecánica del futuro corpus causal regional](future-causal-source-train-review.json)
es una comprobación separada de sus filas TRAIN: conserva 86 de las 95 filas
originales, con 177 de las 193 acepciones; 81 filas siguen idénticas y cinco
conservan sólo sus acepciones admitidas, con procedencia y partición iguales.
Sus cuerpos VALIDATION también se omitieron. La limpieza de esa entrada
regional por sí sola no garantiza que otra entrada de la mezcla futura no
contenga la misma acepción defectuosa. La
[revisión conjunta de TRAIN](future-wide-train-review.json), realizada por otro
agente de IA del proyecto, verificó 15.251 filas de la mezcla futura, incluyendo
86 filas regionales con 177 acepciones y 610 filas genéricas retenidas, todas
idénticas a las entradas curadas admitidas. No quedaron las once glosas
defectuosas conocidas ni equivalentes genéricos de esos registros originales;
las acepciones regionales excluidas también están ausentes. Se omitieron 1.997
cuerpos VALIDATION y no se abrió TEST. No constituye una certificación factual
independiente de todos los datos restantes.

El [helper de revisión conjunta](review_future_wide_train.py) se reproduce con
`--repo "$PWD" --output /tmp/unidad-future-wide-train-review.json`. Usa un
analizador léxico para encontrar el `split` superior sin confundirlo con campos
de procedencia anidados. Se niega a reemplazar una salida existente y verifica
los hashes de entradas curadas, manifiesto y shards antes y después. Esta
revisión tardó 1,07 segundos con 37,81 MiB de RSS; no encola ni entrena.
