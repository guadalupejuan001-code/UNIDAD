# Siguiente lote causal de español: Tatoeba y retención OASST

Lote preparado para una **etapa futura**, sin detener ni modificar el ajuste
conversacional activo. La preparación no entrena parámetros ni demuestra que
UNIDAD haya aprendido estos textos. Este directorio no encola corridas.

El corpus reúne 1.981 documentos nuevos de
[Tatoeba](../new-sources/tatoeba-spanish-next-v1/README.md), con 139.013 frases
independientes, y 2.299 documentos OASST que se repiten sólo para retención.
Tatoeba aporta 708 familias de autor contenidas en este lote: 612 no figuraban
en su lote previo y 96 prolongan familias anteriores, conservando su partición.
Los documentos empaquetan frases con saltos de línea; **no son conversaciones
continuas verificadas**. OASST de retención no cuenta como diversidad nueva.

| Distribución | Tatoeba | Retención OASST | Total |
| --- | ---: | ---: | ---: |
| TRAIN | 1.725 | 2.059 | 3.784 |
| VALIDATION | 256 | 240 | 496 |
| TEST publicado o utilizado | 0 | 0 | 0 |

El archivo final contiene 18.595.269 bytes, dentro del límite de 24 MiB. Se
necesitó un único shard y no se dividieron familias. El conteo de palabras
Unicode del mixer es 968.017 para Tatoeba y 277.063 para OASST; el preparador
original de Tatoeba cuenta 969.442 unidades separadas por espacios. Son métricas
distintas, no dos ampliaciones de datos.

La fuente Tatoeba es el export fijado del 10 de octubre de 2026; las frases
retenidas de este lote tienen fechas de incorporación de 2012 a 2016. La fecha
del export no certifica hechos actualizados ni autoría exclusivamente humana.
Se conservan íntegros el texto, los IDs, las URLs y la licencia por registro.
Consulte la [licencia y atribución](../new-sources/tatoeba-spanish-next-v1/LICENSE-DATA.md),
su [registro de autores](../new-sources/tatoeba-spanish-next-v1/attribution.json)
y la [revisión de condiciones](../new-sources/tatoeba-spanish-next-v1/license-review.json).
Tatoeba usa CC BY 2.0 France por defecto y CC0 en las coincidencias documentadas;
OASST conserva Apache-2.0. No se reasignan a una licencia común.

La guarda histórica conserva [initial-seed.json](initial-seed.json) exactamente,
con SHA `e7b16b6335ae64858b9ea98600948262bbe27200fd325cbba84924ff025af963`.
[wide-history-seed.json](wide-history-seed.json) añade los 4.705 registros
TRAIN/VALIDATION del corpus amplio congelado, sin cambiar ninguna clave vieja.
Añade 4.705 hashes completos UTF-8, 4.705 completos con espacios normalizados
y 58.382 hashes de bloques y líneas de al menos 12 palabras. Conserva todos
los folds de cada hash y los folds de las 1.582 familias del corpus amplio.
La extensión no encontró hashes propios con varios folds; la señal sigue
siendo una lista, sin escoger arbitrariamente un fold cuando los haya.

El [mixer original](../../mix_spanish_public_sources.py) permanece intacto con
SHA `3325d02b789e3fa7ef2a7e18a2b2799e608e9bb54f44b8b400df3dcf97791efc`.
El [helper derivado](../../mix_spanish_public_sources_v2.py) admite esas guardas
adicionales y rechaza cambios de partición de familias existentes. Mantiene la
guarda humanV3 y las anclas OASST congeladas. La preparación previa de Tatoeba
ya había descartado IDs y variantes mecánicas de su lote anterior; las guardas
adicionales de esta mezcla excluyeron cero familias. Esto **no prueba ausencia
de paráfrasis semánticas**, frases cortas comunes ni toda filtración contra TEST.
No se abrió ningún corpus TEST, suite de evaluación, modelo ni tokenizador.

La extensión de la semilla tomó 1,45 s de pared, 0,97 s de CPU y 161,6 MiB de
RSS máximo; leyó 25.357.044 bytes y escribió 16.464.431 bytes incluyendo su
recibo. La mezcla tomó 3,04 s de pared, 1,27 s de CPU y 194,5 MiB de RSS máximo;
leyó 32.520.471 bytes y escribió 18.878.300 bytes incluyendo manifiestos.
Cada ejecución limitó CPU a un núcleo, memoria a 512 MiB y tiempo a 600 s.
La receta permite hasta 128 MiB de lectura más escritura por ejecución;
ambas operaciones reales también quedaron por debajo de 64 MiB. No se
ejecutaron baterías de software ni entrenamiento en esta preparación.

Desde la raíz del repositorio, reproduzca la mezcla en un directorio nuevo:

```sh
python examples/mix_spanish_public_sources_v2.py \
  --recipe examples/learning-repair-20261013/tatoeba-next-mix-v1/recipe.json \
  --output private/reproduce-tatoeba-next-mix-v1
```

Los pins de la [receta](recipe.json) impiden usar fuentes cambiantes. El corpus
debe conservar SHA `93b87039b64c24d4a32cef6f5199fb6fc7c2d16dc6c402bb61fe15320e36a96e`
y su manifiesto de shard
`3da147ecfbbd6741e32f5114da203bd0cbfec484c78573cf7a70503a12220bd1`.
Los tiempos y rutas del manifiesto global variarán al reproducirlo. Los
[recibos de preparación](seed-preparation-receipt.json) y
[manifiestos](manifest.json) conservan las mediciones e identidades originales.
