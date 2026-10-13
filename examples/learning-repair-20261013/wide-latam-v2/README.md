# Español amplio y Latinoamérica: corpus causal v2

Esta mezcla incorpora los datos de Latinoamérica al siguiente ciclo autorizado.
Su preparación dejó intacto el entrenamiento conversacional activo, no creó
una corrida adicional y no modificó los corpus ni paquetes anteriores. La cola
y los entrenamientos efectivos se registran por separado; estos archivos no
demuestran que el modelo ya haya aprendido sus contenidos.

La receta conserva las cuatro fuentes del corpus amplio v1: Gutenberg,
Wikisource, Tatoeba v1 y Wikcionario. Añade tres fuentes de Latinoamérica:

| Fuente incorporada | Datos retenidos | Qué aportan |
| --- | ---: | --- |
| [Prosa regional](../new-sources/latin-american-prose-v1/README.md) | 20 documentos, 47.782 palabras | Explicaciones sobre variedades del español, con revisiones e historia de autoría. |
| [Definiciones regionales](../new-sources/latin-american-spanish-v1/README.md) | 106 entradas, 229 acepciones, 5.486 palabras | Significados de una muestra de términos, con países y usos registrados. |
| [Catálogo regional](../new-sources/latin-american-spanish-v1/integration-recipe.json) | 10.129 registros, 268.652 palabras | Inventario de términos asociados a categorías directas de 19 países; no contiene todos sus significados. |

Los datos regionales incluyen Argentina, Bolivia, Chile, Colombia, Costa Rica,
Cuba, Ecuador, El Salvador, Guatemala, Honduras, México, Nicaragua, Panamá,
Paraguay, Perú, Puerto Rico, República Dominicana, Uruguay y Venezuela.
La [distribución efectivamente retenida](retained-regional-coverage.json)
conserva los recuentos por país y los documentos regionales con su partición.
El artículo sobre español mexicano conserva VALIDATION; no se convirtió en
entrenamiento. Las categorías cubiertas no representan todo el español de
Latinoamérica ni certifican que todos los hablantes usen cada término igual.

Se prepararon dos shards, de como máximo 24 MiB cada uno. Las 10.713 familias
fuente retenidas se asignaron completas y una sola vez, sin cambiar folds.
Cada shard repite las mismas 2.299 anclas OASST —2.059 TRAIN y 240 VALIDATION—
exclusivamente para retención. Esa repetición no cuenta como diversidad nueva.

| Shard | Bytes | TRAIN | VALIDATION | TEST |
| --- | ---: | ---: | ---: | ---: |
| 001 | 25.165.694 | 9.731 | 1.456 | 0 |
| 002 | 5.952.993 | 5.530 | 542 | 0 |

Hay 12.661 registros fuente y 4.598 exposiciones documentales de retención
OASST entre ambos archivos. Las fuentes aportan 1.792.251 palabras según el
conteo Unicode del mixer; las anclas añaden 277.063 por shard. Las palabras de
metadatos no se incluyen como texto objetivo. El shard 001 reúne todas las
fuentes; el 002 contiene familias completas del catálogo y las anclas OASST.
El siguiente lote Tatoeba preparado por separado no forma parte de esta mezcla.

La semilla histórica es el [archivo original exacto](initial-seed.json), con
SHA `e7b16b6335ae64858b9ea98600948262bbe27200fd325cbba84924ff025af963`.
El corpus amplio v1 estaba en cola y aún no había sido entrenado al preparar
v2; no se usó una semilla ampliada que lo marcara como aprendizaje ya ejecutado.
Los corpus fuente originales, sus particiones y el paquete v1 se conservan.

El mixer verifica hashes completos UTF-8 y con espacios normalizados dentro
de la mezcla y contra humanV3; conserva los hashes completos originales y los
parágrafos históricos disponibles en la semilla inicial. Esta última no dispone
de hashes completos normalizados para todos sus registros. Las guardas de
parágrafos comparan bloques y líneas originales de al menos 12 palabras.
No detectan todas las paráfrasis semánticas, frases cortas o sustituciones de
nombres. No se abrió texto TEST ni una suite de evaluación.

La [auditoría de familias](family-registry-audit.json) y las comprobaciones del
mixer verificaron los tres registros léxicos: cero conflictos de fold y cero
alias con identificadores canónicos diferentes. No se renombraron registros.
Se excluyeron siete familias completas por parágrafos iguales entre folds,
conservando las fuentes originales y los motivos en el manifiesto. Son cinco
variantes de «la tercera es la vencida», «diablo» y «diantre». La exclusión de
«diablo» también retira un registro del catálogo: 10.129 retenidos de 10.130.
Las 106 definiciones regionales y los 20 documentos de prosa permanecen íntegros.

Las licencias y atribuciones siguen por registro y fuente. Consulte las
[condiciones de los datos regionales](../new-sources/latin-american-spanish-v1/LICENSE-DATA.md),
las de [prosa regional](../new-sources/latin-american-prose-v1/LICENSE-DATA.md)
y las referencias de fuentes en la [receta](recipe.json). El catálogo usa un
marco de texto determinista escrito para el proyecto alrededor de metadatos
originales: no son ejemplos de conversación generados por el modelo.

La preparación tomó 2,74 s de pared y 2,16 s de CPU, con 260,2 MiB de RSS
máximo. Leyó 38.819.908 bytes y escribió 33.082.066 bytes con los manifiestos:
71.901.974 bytes de E/S. El límite es un núcleo, 512 MiB de memoria, 600 s y
128 MiB de lectura más escritura por operación. La reserva previa de E/S
superaba 64 MiB, por eso este lote usa el límite mayor. La copia pública y la
verificación de metadatos se midieron aparte en su [recibo](delivery-receipt.json).
El código recibió lectura estática y comprobación sintáctica; la preparación
no cargó Torch, modelos ni tokenizadores, y no ejecutó una batería de software.

Reproducción desde la raíz del repositorio, siempre en un directorio nuevo:

```sh
python examples/mix_wide_latam_spanish.py \
  --recipe examples/learning-repair-20261013/wide-latam-v2/recipe.json \
  --output private/reproduce-wide-latam-v2
```

El [helper congelado](mix_wide_latam_spanish.frozen.py) conserva los bytes de
esta versión; el comando usa su copia canónica en `examples` para resolver las
rutas del repositorio. Los dos mixers anteriores permanecen sin modificaciones.
Las configuraciones de base adjuntas son copias exactas de las ya publicadas,
no prueba de otro modelo entrenado. El SFT natural regional se prepara y registra
por separado, sin convertir el catálogo causal en respuestas supervisadas.

| Identidad | SHA-256 |
| --- | --- |
| Corpus 001 | `df6401e9525838f26d73759a9933be1402ee0fd3500a3e03965da40e588b387f` |
| Corpus 002 | `e5e974f8fae72a4ae73c9c3716f3e016a2c7e603d89596653c3019a838db784d` |
| Manifiesto 001 | `f5c0873c7a3f35e32001a00f8925bc5735f50d13fb4b43e16d5cf9f635ada69e` |
| Manifiesto 002 | `8edeb15c276cab4e4c16d91da369252eee743d79d9de8be61d94d7ec23b70d7b` |
| Receta | `c2aa168113a37b7d5c32373c8fbc1a5f9f21712f954e0a5ecb6c72420f1085df` |
| Helper | `69bfaf2a89ebd8b05a3792194625fa7f62863ed91a11f966043a2d4092531e89` |

Los hashes de corpus y manifiestos de shard deben coincidir al reproducirlos;
las mediciones del manifiesto global cambian con la ejecución. Preparar archivos,
reducir pérdida o tener checkpoints no acredita conversación competente.
