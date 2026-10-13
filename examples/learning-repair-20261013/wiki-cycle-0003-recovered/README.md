# Corpus recuperado de español: wiki-cycle-0003

Copias verificadas byte a byte del corpus que empezó a entrenar UNIDAD el
13 de octubre de 2026 después de recuperar las descargas HTTP 502.

Incluye 2.553 registros nuevos de 2.545 artículos de Wikipedia en español,
snapshot `20231101.es` de la revisión
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, y 2.299 registros OASST conocidos
de repaso. TRAIN 4.325, VALIDATION 527, TEST 0. Las familias y variantes no
cambian de partición; no se vuelve a ajustar el tokenizador.

`manifest.json` identifica fuentes, selección de prosa, exclusiones y hashes;
`source-pages.json` conserva las 58 páginas de origen y sus recibos. Cada
registro conserva procedencia y atribución. La tarjeta del snapshot declara
CC-BY-SA-3.0/GFDL y la documentación actual de Wikimedia declara
CC-BY-SA-4.0/GFDL; se conservan ambas declaraciones en el manifiesto. OASST
declara Apache-2.0. No se otorga una nueva licencia única al conjunto.

Las transformaciones son selección de prosa, partición y ventanas, descritas
por los preparadores existentes, no generación de nuevos artículos. Consulta
[condiciones y recibos de las fuentes](../collector/source-license-receipt.json).
Se mantienen títulos, URLs y atribución por el historial de los artículos.

SHA256 de `causal-spanish.train-validation.jsonl`:
`4dcd3a9e1f8573ff5030f37d7a0cd22768bd64d3dbdeb706594a8d7dbedb8483`.
La presencia de este corpus no demuestra que todos sus contenidos se hayan
consumido o aprendido. La calidad conversacional se mide por respuestas reales.

[Recuperación, límites e inicio de entrenamiento](../../../docs/learning-repair-20261013/download-recovery/README.md).
