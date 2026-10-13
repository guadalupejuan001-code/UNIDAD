# Entrenamiento con datos tokenizados en disco

La revisión del 13 de octubre encontró un bloqueo concreto para ampliar la base
lingüística: el entrenador retenía el corpus completo y todas sus ventanas como
objetos Python. El nuevo modo `data_loading="mmap"` valida y recorre los registros,
ajusta BPE sólo con TRAIN y guarda las ventanas en disco. Los lotes leen sus
ventanas mediante índices mmap. GNOSIS lo selecciona en sus nuevas etapas.

## Por qué esta intervención

La prioridad del creador sigue siendo conversación competente en español con
pesos propios. La última comparación reservada dio C7 0/40 respuestas completas,
0/10 diálogos completos y cuatro críticas; el ajuste parcial128 también dio
0/40 y ocho críticas. Los experimentos anteriores mostraron aprendizaje de
ejemplos conocidos sin generalización suficiente a formulaciones nuevas.
Los [resultados conservados](brain-nerva-20261013/reserved-comparison.json)
no justifican repetir el mismo ajuste ni promover otro candidato.

Ampliar y diversificar la base lingüística requiere manejar más datos. Este
cambio resuelve su retención completa en RAM y conserva las reglas actuales
de aprendizaje. No cambia arquitectura, inferencia ni pesos de C7. Tampoco
entrena un nuevo modelo conversacional ni demuestra que esa hipótesis de datos
sea suficiente para resolver la conversación.

## Contrato y comprobaciones

- Texto y SFT comparten el mismo generador de ventanas en memoria y en disco.
  Conserva orden, desplazamiento causal, máscaras, supervisión de la última
  respuesta, contexto solapado y un EOS por registro.
- La validación conserva IDs únicos, familias en una sola partición y controles
  de procedencia. Texto conserva también el rechazo global de duplicados exactos.
  BPE sólo recibe TRAIN. TEST no se tokeniza al entrenar ni al exportar el snapshot.
- La identidad de caché incluye corpus, tokenizador, objetivo, contexto, ventanas
  y política de partición. Manifiesto, datos e índices se verifican por hashes,
  tamaños y límites. La escritura publica un directorio completo y no reemplaza
  una caché dañada silenciosamente.
- Una caché ausente se reconstruye a partir de los mismos datos; los checkpoints
  siguen siendo la fuente de pesos, Adam, scheduler, RNG y orden pendiente.
  Una configuración antigua sin `data_loading` conserva el camino anterior.
- Las pruebas comparan pesos, optimizador, pérdidas de validación y RNG exactos,
  además de ventanas y supervisión. La evaluación explícita `finalize` puede
  incluir TEST en una caché separada; no interviene en la selección del candidato.

La [guía de uso](language-pretraining.md#corpus-y-ventanas-desde-disco) describe
configuración, reconstrucción y restricciones. Esta primera versión admite
muestreo uniforme; rechaza pesos de procedencia y muestreo adaptativo en mmap.
Los IDs, familias, hashes de texto, vocabulario y orden aún crecen con el corpus.
Las páginas leídas del mapa también pueden permanecer residentes; no se promete
RAM constante. GNOSIS mantiene una admisión conservadora y cuenta almacenamiento
adicional de caché.

La verificación final local aprobó **462 pruebas del proyecto y 12 de ejemplos**,
incluidas 21 nuevas de corpus/caché e integración. El
[recibo de software](tokenized-data-20261013/software-verification.json)
identifica fuentes, comandos y registros. Se comprobó SFT desde un artefacto
propio, con nuevo Adam y tokenizador heredado, y reanudación exacta con BF16 y
microbatches. La revisión adicional del código corresponde a agentes de IA del
proyecto, no a una auditoría externa.

## Evidencia reproducible

El [runner de medición](../examples/benchmark_tokenized_data.py) usa corpus
ficticios, un tokenizador fijo ajustado sólo a TRAIN, procesos independientes y
orden alternado. Compara todos los tokens y objetivos de TRAIN/VALIDATION mediante
SHA-256. No carga un Transformer ni importa PyTorch en los procesos medidos;
el RSS corresponde a preparación/lectura de datos, no al entrenamiento completo.

El [informe de memoria](tokenized-data-20261013/benchmark.json) conserva tamaños,
repeticiones, tiempos, picos RSS, hashes y versiones. La primera preparación
escribe y comprueba la caché; el coste de reutilizarla se mide por separado.

Medianas de tres procesos por modo, tras leer y verificar todas las ventanas:

| Documentos | Ventanas | RAM memoria | RAM mmap inicial | RAM mmap reutilizado |
| --- | ---: | ---: | ---: | ---: |
| 200 | 927 | 35,64 MiB | 28,61 MiB | 27,45 MiB |
| 2.000 | 9.418 | 120,96 MiB | 43,78 MiB | 42,37 MiB |

Con 2.000 documentos, el pico RSS bajó un **63,81 %** al preparar mmap y un
64,97 % al reutilizarlo. La preparación inicial tardó 1,738 s frente a 1,238 s
en memoria; reutilizar la caché tardó 0,293 s. No se atribuye una mejora de
velocidad a la primera preparación. Los corpus tienen entre 2 y 4 KiB de texto
por documento, vocabulario de 512 y contexto de 256. Estas mediciones no estiman
el pico de un Transformer ni una base pública grande.

```sh
python examples/benchmark_tokenized_data.py --records 200 2000 \
  --min-text-bytes 2048 --max-text-bytes 4096 --repeats 3 \
  --output /tmp/unidad-tokenized-reproduction.json
```

El archivo de salida debe ser nuevo. El runner conserva la serialización del
tokenizador para identificar y reproducir los mismos IDs.

La [regresión de referencia](tokenized-data-20261013/reference-regression.json)
reproduce cuatro respuestas archivadas de C7, usando cada respuesta realmente
generada como historia del siguiente turno. Los cuatro archivos del artefacto
conservan sus hashes publicados. El
[script](tokenized-data-20261013/verify_reference.py) permite repetirla; es una
regresión de conservación, no una aceptación nueva ni una mejora conversacional.

## Qué continúa pendiente

La base pública más amplia debe prepararse con procedencia, derechos de uso y
particiones conservadas. Después necesita entrenamiento propio con recursos
declarados, comprobación de retención y diálogos nuevos. La aceptación permanece
en 36/40 respuestas completas, 8/10 diálogos completos y cero críticas o errores
técnicos. GPU, conversación competitiva y programación autónoma siguen pendientes.
Los presupuestos históricos permanecen cerrados; este cambio no deja ningún
proceso de entrenamiento activo.
