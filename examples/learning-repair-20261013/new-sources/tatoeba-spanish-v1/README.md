# Tatoeba español: shard causal con atribución

Fuente oficial: [Detailed Sentences](https://tatoeba.org/en/downloads), export
`spa_sentences_detailed.tsv.bz2` del 10 de octubre de 2026. Se conservaron
97.228 frases en 1.432 documentos: **1.223 TRAIN y 209 VALIDATION, cero TEST**.
Cada documento reúne frases de un mismo usuario público/licencia mediante
saltos de línea, sin modificar el texto original. Las frases independientes
no se presentan como una conversación ni una narración continua.

`tatoeba-spa.train-validation.jsonl` tiene 10.919.383 bytes y SHA-256
`5b4c7e6f46b667fc41c0088f0265f09a0d81aa0ff5d178f7401f276bc6014e79`.
Su texto, sin metadatos, ocupa 4.130.011 bytes. El manifiesto tiene SHA-256
`ac62add0f0e386678236f61b4f3bb7677968ac7de720f5d56be9493a3b171d28`.
El corpus usa el esquema causal nativo; no se cargó modelo ni tokenizador.

| Fold | Documentos | Frases | Familias de usuario | Bytes de texto |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 1.223 | 83.121 | 455 | 3.461.807 |
| VALIDATION | 209 | 14.107 | 52 | 668.204 |

Se procesaron exactamente 100.000 filas TSV: 99.963 del prefijo del archivo
detallado y 37 del archivo CC0 oficial. El export completo se descargó y se
conserva comprimido en el archivo privado histórico; no se distribuye aquí ni
se recorrió completo. Este shard tiene sesgo de
orden: las fechas de alta retenidas son principalmente 2007–2012 y hay una
fecha legacy de 1970. El snapshot publicado en 2026 no convierte esas frases
en hechos actualizados en 2026 ni en una muestra exhaustiva del español actual.

La partición se calcula con `assign_split` nativo sobre una familia estable del
username exportado, normalizado NFKC/casefold y convertido a SHA. No se movieron
familias para equilibrar folds. Las 2.333 frases asignadas a TEST se omitieron;
no existe un archivo TEST de enseñanza. Se retiraron 344 duplicados de variante
y 58 clases de variantes que aparecían en diferentes folds. La clave de
deduplicación usa NFKC, casefold, espacios y eliminación de puntuación Unicode;
esto es una guarda conservadora, no un análisis de equivalencia semántica.
Las claves no reemplazan las frases originales.

La licencia por defecto del export es [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/),
según las condiciones y la página oficial de descargas. Cada registro incluye
username, perfil público, IDs de frases y URLs originales para atribución y
consulta de su procedencia. El archivo `attribution.json` contiene
los 507 usuarios exportados, folds y licencias; SHA-256
`214b60d907f70138b6f5fb083e0370b03068fcb560c80b424bee5ae0da8978fd`.
Las coincidencias exactas de ID y texto en `spa_sentences_CC0.tsv.bz2` conservan
la etiqueta [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
El username identifica al propietario consignado por Tatoeba; no se certifica
independientemente que sea el creador original. Una frase sin atribución se
excluiría salvo coincidencia exacta CC0; no hubo huérfanas aceptadas en este shard.

`license-review.json` conserva las citas oficiales, formato de columnas y
límites de procedencia. SHA-256:
`71e4ad7234cd1362303404b7602e958c4e604b03ae5a465a2cb045a47e9328a4`.
Las condiciones de Tatoeba no verifican las competencias de participantes ni
la veracidad de sus afirmaciones. Este corpus comunitario no se certifica como
exclusivamente humano ni se usa como prueba de conocimientos factuales.

El archivo detallado completo tiene 9.502.065 bytes, ETag `6ac9db4c-90fd71` y
SHA-256 `3141acd2354ecbb65199268ec222f680a232ca494129ba77b53a16b3890a11e0`.
El CC0 tiene 2.266 bytes, ETag `6ac9db9b-8da` y SHA-256
`e8cecab4e084fa4d9843f66c1dab90c5747057e0bc63a758c406235297b636c7`.
Sus URLs, fechas y respuestas HTTP quedan en `source-receipts/*.receipt.json`.

La ejecución final consumió 4,252 s de pared, 1,694 s de CPU y 132,793 MiB RSS.
Se impusieron CPU 1, 600 s, espacio de direcciones 512 MiB, disco 256 MiB,
red 64 MiB, corpus 24 MiB y 100.000 filas. El primer intento fue rechazado antes
de publicar porque el SHA del CC0 pequeño se calculó antes de vaciar el buffer
de escritura. Se conservaron fuente y recibo fallidos, se añadió flush/fsync y
se volvió a descargar únicamente el CC0; la fuente detallada quedó intacta.
`first-attempt-rejected.json` conserva ese hecho. No afectó entrenamientos.

`prepare_tatoeba.py` conserva las reglas y la ingesta ejecutada. Rechaza
reemplazar `prepared-v1`; las fuentes oficiales están fijadas por ETag/longitud
y luego por SHA. El publicador comprueba fuente, código y esquema antes de
renombrar staging. Los documentos se validaron con el esquema causal nativo.
No se ejecutaron pruebas de software, inferencia ni entrenamiento. No se leyó
el historial personal ni suites de evaluación. El curador debe deduplicar este
corpus contra los otros corpus causales antes de una nueva etapa; esta fuente
no modifica la corrida ni el corpus activos.

Las rutas privadas del manifiesto son históricas de la ingesta.
`delivery-receipt.json` mapea los archivos publicados sin cambiar esos recibos.
El raw completo y la evidencia del primer intento rechazado permanecen privados;
el corpus TRAIN/VALIDATION preparado se conserva aquí byte por byte. La
reconstrucción de la ingesta original requiere el raw histórico: las URLs de
Tatoeba se actualizan y el helper rechaza un ETag diferente. El entrenamiento
puede reutilizar directamente este corpus fijado sin volver a descargar nada.
