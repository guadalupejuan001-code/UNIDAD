# Español regional curado: datos causales futuros v3

**PREPARADO, NO ENCOLADO.** Este paquete contiene datos para una etapa futura,
no pesos entrenados. No cambió fuentes anteriores, la corrida activa, código
conversacional, configuraciones ni la cola. La revisión es de IA del proyecto,
no una validación semántica humana independiente.

El [helper reproducible](curate_future_causal.py) usa sólo la biblioteca estándar
y la [receta fijada por SHA](recipe.json). Parte del corpus regional original
`d27c633d6166f42d7fe5d59d865b5a8a3ba67c6d3838035d059178184d1d8f19`.
Retira las nueve acepciones de la revisión previa y las dos acepciones botánicas
adicionales rechazadas en la revisión final: algarrobilla 1.3 y caraota 1.1.
También retira siete acepciones exactas con abreviaturas de uso opacas ya
excluidas en el manifiesto SFT v3. Abajeño 1.1 ya pertenecía a las nueve, por lo
que la unión contiene **18 acepciones**, sin doble conteo.

La [lista completa de exclusiones](definitions/excluded-senses.json) conserva
el significado original, identidad de acepción y fuente de la revisión. El
corpus resultante tiene **96 entradas y 211 acepciones**, frente a las 106 y
229 originales: 86 TRAIN y 10 VALIDATION. Diez entradas quedan sin acepciones
y se omiten; seis entradas conservadas se reconstruyen; las otras 90 mantienen
sus bytes originales. Se preservan literalmente las acepciones restantes,
ejemplos, países, advertencias de uso, licencias, historial, revisiones y folds.
El campo estructurado original sigue siendo `source.regionalSenses`.

La comprobación del primer candidato detectó que la fuente genérica antigua
reintroducía el significado defectuoso de abajeño. El
[recibo del hallazgo](preflight-wide-v3-defect-discovery.json) y sus manifiestos
guardan esa evidencia. Se preparó además un subconjunto genérico de Wikcionario:
se omite sólo `wiktionary:es:12076:6021556`, con el mismo pageId y revisionId que
la acepción defectuosa. Su esquema carece de acepciones estructuradas, por lo
que se retira el registro completo. Contenía un único significado y ningún
ejemplo; no se descartaron acepciones válidas adicionales. Las otras **718
filas genéricas conservan sus bytes exactos**, según el
[recibo genérico](definitions/generic-excluded-records.json).

No se excluyen sinónimos válidos por coincidencia de una cadena de texto:
«malograr el funcionamiento o propósito de algo» permanece en joder, una
identidad de fuente distinta de la acepción chingar excluida por uso opaco.
El catálogo de países sigue siendo metadatos de clasificación; no se convierte
en una definición ni se elimina una clasificación verdadera por retirar una
glosa defectuosa.

Los seis casos de país/acepción de guagua, chamba y china pasan la
[comprobación de datos](definitions/golden-country-senses-receipt.json).
No se abrieron cuerpos TEST o reservados, ni se cargaron Torch, modelos o
tokenizadores. Preparación: un núcleo, 512 MiB, 120 s y hasta 2 MiB de entrada
y salida respectivamente; observada en 0,023 s y 20,6 MiB de RSS máximo.
La [reproducción en una salida nueva](reproduction-receipt.json) conservó los
nueve archivos deterministas byte por byte; el manifiesto de mediciones varía.

Reproducción desde la raíz, usando siempre una salida nueva:

```sh
timeout 120 python examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3/curate_future_causal.py \
  --recipe examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3/recipe.json \
  --output private/reproduce-latin-american-spanish-curated-v3
```

La mezcla causal final está en [wide-latam-v3](../../wide-latam-v3/README.md).
Sus recibos fijan estos archivos ya preparados; reproducir la curación en otra
ruta no modifica automáticamente los hashes y rutas de la receta de mezcla.
Las fuentes originales y sus [condiciones de licencia](../latin-american-spanish-v1/LICENSE-DATA.md)
se conservan, al igual que las [referencias del Wikcionario genérico](../wiktionary-spanish-v1/README.md).
CC-BY-SA-4.0 y la atribución individual permanecen en cada fila. Este conjunto
finito no acredita cobertura de todo el español ni calidad conversacional.

| Archivo causal | SHA-256 |
| --- | --- |
| Regional | `9c0057f5dc5d00007d971605615bbb1a01c592a6b585e5937a4af62c0f29199b` |
| Genérico | `ea0c44aec988e22ba4495cb059b7ea558c63a10d7ac8cf2c83946194646fd5f3` |
