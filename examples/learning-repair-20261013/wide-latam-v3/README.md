# Mezcla causal futura de español y Latinoamérica v3

**PREPARADA, NO ENCOLADA.** Estos archivos no son pesos entrenados ni evidencia
de aprendizaje ejecutado. No se alteró la corrida activa, la cola, ninguna
configuración o corpus histórico. Los paquetes v1 y v2 permanecen intactos.

La [receta](recipe.json) deriva de v2. Cambia únicamente los dos pares de
rutas/hashes de corpus y procedencia de Wikcionario genérico y definiciones
regionales a los [datos curados](../new-sources/latin-american-spanish-curated-v3/README.md),
y corrige el recuento descriptivo regional a 96 entradas y 211 acepciones.
Mantiene exactamente la referencia de semilla inicial, registries, fuentes
restantes, licencias, límites y anclas de retención. No añade Tatoeba-next,
configuraciones de modelo o entrenamiento ni invoca una etapa de aprendizaje.

La curación retira 11 significados defectuosos y siete acepciones adicionales
con códigos de uso opacos: unión de 18 acepciones regionales. Además omite el
registro genérico de abajeño con idéntica página/revisión para impedir que esa
fuente vuelva a introducir el significado rechazado. Cada fila restante del
subconjunto genérico conserva sus bytes originales. Los significados válidos
de otras identidades, como el de joder, permanecen.

La [QA de toda la mezcla](qa-receipt.json) comprobó **cero significados
conocidos excluidos retenidos**, tanto por identidad de fuente como por las
11 cadenas defectuosas exactas en el texto de todos los shards. Cada registro
emitido coincide byte por byte con su fuente preparada fijada; cero cambios
de fold. Esta comprobación es mecánica y de IA del proyecto, no una nueva
evaluación conversacional ni una aprobación semántica humana del resto del
corpus. Las revisiones y licencias originales permanecen por registro.

| Shard | Bytes | TRAIN | VALIDATION | TEST |
| --- | ---: | ---: | ---: | ---: |
| 001 | 25.165.227 | 9.746 | 1.457 | 0 |
| 002 | 5.932.628 | 5.505 | 540 | 0 |

Hay 12.650 filas fuente nuevas y 4.598 exposiciones de retención OASST: las
mismas 2.299 anclas se repiten en cada shard y no cuentan como diversidad
nueva. Las 10.713 familias fuente retenidas se asignan completas y una sola
vez. Los guards del mixer excluyen las mismas siete familias por duplicados
entre folds y no posponen familias por tamaño. El catálogo conserva sus
clasificaciones por país; retirar una definición no convierte su clasificación
verdadera en un hecho defectuoso. La cobertura sigue siendo una muestra finita.

El [manifiesto](manifest.json) conserva hashes, procedencia, familias, guardas,
costes y mediciones. La preparación usó el mixer existente sin cambios:
un núcleo, 512 MiB, 600 s y 128 MiB de lectura más escritura. Se observaron
2,97 s de pared, 2,48 s de CPU y 260,0 MiB de RSS máximo; 38.675.439 bytes
leídos y 33.061.015 escritos. La QA posterior tomó 0,51 s y 68,4 MiB de RSS,
con límite separado de un núcleo, 512 MiB, 120 s y 128 MiB leídos. No se
abrieron cuerpos TEST o reservados ni modelos, tokenizadores o Torch.

Reproducción desde la raíz, siempre con salida nueva:

```sh
timeout 600 python examples/mix_wide_latam_spanish.py \
  --recipe examples/learning-repair-20261013/wide-latam-v3/recipe.json \
  --output private/reproduce-wide-latam-v3
```

El [helper congelado](mix_wide_latam_spanish.frozen.py) es idéntico al mixer
canónico usado. Los hashes de corpus y manifiestos de shard deben coincidir;
el manifiesto global incorpora mediciones de ejecución variables. Para volver
a comprobar las fuentes y la mezcla:

```sh
timeout 120 python examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3/verify_future_causal.py \
  --curation-recipe examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3/recipe.json \
  --curated-output examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3/definitions \
  --wide examples/learning-repair-20261013/wide-latam-v3 \
  --receipt private/reproduce-wide-latam-v3-qa.json
```

| Identidad | SHA-256 |
| --- | --- |
| Corpus 001 | `81c414480d0baf36b7784f808ba8d4759a6b7ef706eeaabdcb74d6e919cb8a81` |
| Corpus 002 | `411c0a8033deca79e471918db61bdb5ba0fff72935ccfd4248ddfe046d7dcfca` |
| Receta | `cf232beb66eb82a67e5295fe9b30ef1f7c7e0e0dd59e1c0d1f7b116554667aff` |
| Mixer | `69bfaf2a89ebd8b05a3792194625fa7f62863ed91a11f966043a2d4092531e89` |

La aceptación conversacional y el entrenamiento real deben medirse después,
mediante una etapa autorizada y registrada por separado. Preparar este paquete
no declara conseguido ese objetivo.
