Este lote añade **prosa explicativa sobre variedades de español de Latinoamérica**, obtenida realmente de la API primaria de Wikipedia y congelada por revisión. Es texto causal, no diálogos SFT, un catálogo completo, ni una capacidad regional ya aprendida por UNIDAD. Su incorporación al ciclo de entrenamiento corresponde al supervisor después de la revisión del lote; este preparador no cambia colas ni procesos.

Contiene 20 artículos, 19 registros TRAIN y 1 VALIDATION, sin TEST publicado ni utilizado: 42,066 y 5,716 palabras, respectivamente. El JSONL ocupa 336,685 bytes. Conserva vocabulario regional, voseo/tuteo/ustedeo, características morfosintácticas y variación sociolingüística descritos por las fuentes; no homogeneiza países, regiones, generaciones o hablantes.

| Familia fija antes de chunking | Artículos | Partición |
| --- | ---: | --- |
| Rioplatense: Argentina/Uruguay | 1 | TRAIN |
| Chilena | 1 | TRAIN |
| Caribe: Cuba, República Dominicana, Puerto Rico y Venezuela | 4 | TRAIN |
| Andina: andino, Perú, Bolivia, Ecuador y Colombia | 5 | TRAIN |
| Centroamérica: centroamericano, Panamá, Costa Rica, Honduras, Guatemala, Nicaragua y El Salvador | 7 | TRAIN |
| Paraguaya | 1 | TRAIN |
| Mexicana | 1 | VALIDATION |

La agrupación protege artículos cercanos y sus variantes antes de separar fragmentos. Es una agrupación de estudio: Colombia y Venezuela incluyen variedades fuera de la macrorregión asignada. México cae en VALIDATION por `assign_split("wikipedia-es-dialect-family:mexicano")`; no se movió su partición después de leer los datos. Las siete familias conservaron una única partición. Las menciones de otras regiones dentro de un artículo no constituyen aislamiento semántico perfecto.

Se consultaron 20 títulos predefinidos: todos existían, sin redirecciones ni títulos inventados. Se guardaron 25 respuestas HTTP de la fuente, incluidas relecturas de los IDs exactos de revisión; 1,216,068 bytes descargados. La consulta CC BY-SA añade 36.018 bytes, fuera de ese total de respuestas API. Las revisiones llegan hasta 2026-10-12T11:15:53Z. La ventana entre el primer y último recibo API duró 62,798 segundos; es una ventana de recibos, no una medición de todo el proceso.

Se preservó el primer intento fallido: su limpiador confundía referencias autocerradas con referencias pareadas y absorbía un cierre de tabla venezolano. Se corrigió el orden de eliminación de referencias y se reprocesaron los mismos RAW verificados por hash, sin descargar de nuevo. La inspección adicional retiró 25 párrafos con huecos visibles causados por plantillas y 20 con marcado no resuelto; no se inventaron frases de sustitución. La preparación final duró 3.063 s y usó 67,908 KiB de RSS máximo, CPU 1 y límite de RAM de 512 MiB. La revisión independiente detectó además un pie de imagen con cierre de enlace residual; el filtro genérico ahora retira todo párrafo con ese marcado, sin reescribir el contenido. Los intentos previos y sus fuentes siguen privados e inmutables.

La deduplicación compara texto completo y párrafos o líneas normalizados de al menos 12 palabras con los corpus TRAIN/VALIDATION inicial y wide español congelados. No consultó TEST, evaluación reservada, tokenizadores, modelos ni mensajes del propietario. No encontró conflicto de artículo o de párrafo que exigiera excluir una macrofamilia; esa política está registrada. Detecta coincidencias exactas, no todas las paráfrasis ni el solapamiento semántico entre regiones.

- Corpus SHA-256: `7fee355119358f89b2f4d3a7a7e5b51adcbc47135f36b9fa8c836305678ca5cf`.
- Contenido SHA-256: `e141a6eb7ca904e3e7dfc164a87c65fd9ccd338bdb4851584896d417b3ac7de0`.
- Preparador congelado SHA-256: `56192bde4c4bf750011a28343e5af970d97785c1a32e8c4539c630f49a429d98`.

`manifest.json` conserva títulos, páginas, revisiones, fechas, hashes, URL oldid, historial de autores, límites y fuentes comparadas. `primary-api-receipts.json` publica los metadatos de solicitudes efectivas; los cuerpos HTTP con referencias/citas y el historial de intentos están en `private/learning-repair-20261013/spanish-expansion/` y no se distribuyen. `LICENSE-DATA.md` identifica condiciones y cambios.

Desde la raíz del repositorio, con el proyecto instalado, el preparador ejecuta una nueva ingesta acotada si se usa un directorio de salida nuevo. Las revisiones actuales pueden haber cambiado: para reproducir esta entrega, use el corpus byte a byte y sus hashes congelados, o el replay de las respuestas originales privadas si están disponibles:

```bash
python examples/prepare_latin_american_prose.py \
  --output private/latin-american-prose-replay-nuevo \
  --replay-receipts private/learning-repair-20261013/spanish-expansion/latin-american-prose-v1-attempt-001 \
  --seed examples/learning-repair-20261013/causal-base/causal-spanish.train-validation.jsonl \
  --seed examples/learning-repair-20261013/wide-spanish-v1/wide-spanish.shard-001.train-validation.jsonl
```

Wikipedia tiene errores, contenidos envejecidos y juicios sociales que esta ingesta no verifica exhaustivamente. Al descartar plantillas/tablas se pierde parte de la notación fonética y el léxico. El lote no acredita conversación natural regional ni mejora del modelo: eso requiere entrenamiento efectivo y respuestas nuevas comparables.
