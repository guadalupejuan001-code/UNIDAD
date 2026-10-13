# Español nuevo de Tatoeba: siguiente lote congelado

Este lote añade **139.013 frases** que no estaban en el primer corpus publicado,
con 969.442 palabras separadas por espacios. Proceden de las siguientes
150.000 filas del mismo export oficial español: filas **99.964–249.963**,
contadas desde uno. No se repiten frases anteriores para inflar estos números.
Las frases independientes se concatenan por username público y licencia;
no se presentan como conversaciones continuas.

| Fold | Documentos | Frases | Familias de usuario | Palabras | Bytes de texto |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 1.725 | 119.949 | 628 | 840.442 | 4.858.262 |
| VALIDATION | 256 | 19.064 | 80 | 129.000 | 751.537 |
| TEST | 0 | 0 | 0 | 0 | 0 |

[tatoeba-spa-next.train-validation.jsonl](tatoeba-spa-next.train-validation.jsonl)
tiene 15.449.838 bytes y SHA-256
`16f64e542dd8214f9d69d62e7f82b22a54d288f1280cf4aa22e3665d89e9b006`.
[manifest.json](manifest.json) conserva distribución, procedencia, límites,
hashes, filtros y uso real de recursos. El lote está preparado para una etapa
futura; esta ingesta no cambió el corpus activo ni entrenó pesos.

La familia canónica es el mismo hash del username NFKC/casefold que en v1, y
el mismo `assign_split` nativo conserva su fold. Hay 708 familias: 96 ya
aparecían en v1 y 612 son nuevas. No se trasladó ninguna familia para equilibrar
los folds. Se leyeron 97.228 frases TRAIN/VALIDATION anteriores sólo como semilla
de deduplicación; no se copiaron a este lote. Las 9.588 filas nuevas asignadas
a TEST se omitieron antes de normalizar, agrupar o publicar su texto.
No se abrió ningún archivo TEST histórico ni suite de evaluación.

Se descartaron 687 coincidencias de ID o variantes mecánicas del lote anterior,
249 frases nuevas en clases que cruzaban TRAIN/VALIDATION y 463 variantes
internas adicionales. Las claves usan Unicode NFKC, casefold, espacios,
puntuación, diacríticos y números decimales en frases de al menos cuatro
palabras. Estas guardas conservadoras pueden retirar textos no idénticos;
no detectan todas las paráfrasis semánticas ni sustituciones de nombres.
La deduplicación con otras fuentes y variantes no inspeccionadas de TEST
no se afirma como resuelta exhaustivamente.

La fecha del export es 10 de octubre de 2026; las frases retenidas fueron dadas
de alta principalmente en **2012–2016**. Un snapshot reciente no convierte
las afirmaciones en información actual de 2026. Las condiciones oficiales
no certifican contenido exclusivamente humano ni verdad factual.
[LICENSE-DATA.md](LICENSE-DATA.md) y [attribution.json](attribution.json)
conservan las licencias y 708 créditos públicos.

El export detallado íntegro archivado tiene 9.502.065 bytes y SHA-256
`3141acd2354ecbb65199268ec222f680a232ca494129ba77b53a16b3890a11e0`.
Se verificaron sus bytes antes y después. El archivo CC0 tiene 2.266 bytes y
SHA-256 `e8cecab4e084fa4d9843f66c1dab90c5747057e0bc63a758c406235297b636c7`.
Las respuestas HTTP históricas originales están en [source-receipts](source-receipts).
No se realizó otra descarga: el export oficial completo ya existía en el
archivo privado. El resto de filas sigue diferido y no se contó; el siguiente
offset detallado es **249.963**. No se afirma haber procesado todo el export.

La preparación real consumió 6,128 s de pared, 4,872 s de CPU y 220,457 MiB RSS,
con cero bytes de red nuevos. Se fijaron CPU 1, 600 s, 512 MiB de espacio de
direcciones, 20 MiB de corpus y 128 MiB de disco de salida. Se usó `nice -n 19`
y afinidad de una CPU. No se cargaron Torch, modelos ni tokenizadores, ni se
ejecutaron pruebas de software, inferencia o entrenamiento.

[prepare_tatoeba_next_spanish.py](prepare_tatoeba_next_spanish.py) es una copia
exacta del helper ejecutado fuera de las fuentes ML congeladas.
[recipe.json](recipe.json) da el comando desde la raíz del repositorio.
Requiere el export histórico fijado y el primer corpus publicado; rechaza
fuentes cambiadas y reemplazar una salida existente. Puede reutilizarse el
JSONL fijado directamente para una etapa causal. Las URLs actuales se actualizan
y no garantizan reconstruir el snapshot antiguo. El helper sólo prepara este
lote: no encola ni automatiza otros segmentos.

La preparación y la revisión son trabajo de IA del proyecto; no son una
evaluación humana independiente ni demuestran aprendizaje o conversación.
