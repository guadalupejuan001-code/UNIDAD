# Español latinoamericano: catálogo y acepciones de Wikcionario

Adquisición real del 13 de octubre de 2026 mediante la
[API primaria de Wikcionario](https://es.wiktionary.org/w/api.php). Se agotaron
las continuaciones de **19 categorías de países**: 22.555 membresías y **11.241
IDs de entradas distintas**. El catálogo es completo para los miembros directos
del espacio principal de esas categorías durante las consultas; no abarca sus
subcategorías, todos los usos regionales ni todo el español latinoamericano.
Las consultas no forman una instantánea global atómica. La fecha de pertenencia
en los metadatos indica incorporación a una categoría, no revisión de una glosa.

Hay dos corpus causales distintos, ambos listos para incorporación posterior:

| Corpus | TRAIN | VALIDATION | TEST en el input | Bytes | Palabras |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clasificación del catálogo | 9.005 | 1.125 | 0 | 7.823.002 | 268.661 |
| Acepciones regionales limpias | 95 | 11 | 0 | 210.116 | 5.723 |

El [catálogo causal](catalogue/latin-american-catalogue.train-validation.jsonl)
agrupa todos los países de cada entrada. Su texto sólo afirma que Wikcionario
la registra en esas categorías y explica que la clasificación no especifica
acepciones ni usos exclusivos. Este formato fue escrito por el proyecto y
aplicado de forma determinista, sin generar datos con otro modelo. Sus
268.661 palabras provienen de un formato repetido; no representan esa cantidad
de conocimiento nuevo ni diversidad conversacional. SHA256:
`c7a4315d9cc652cd123c3c25fef190fd89a1750165541487ac6ab5577fb20ee5`.

El [corpus de acepciones](definitions/latin-american-wiktionary.train-validation.jsonl)
contiene **229 sentidos explícitamente regionales**, extraídos de una muestra
estratificada de 240 páginas. Se recibieron 120 páginas útiles; 106 quedan en
TRAIN/VALIDATION y 14 en TEST privado. Conserva etimología, número de acepción,
categoría gramatical, glosa, región textual, notas y registro de uso. SHA256:
`d27c633d6166f42d7fe5d59d865b5a8a3ba67c6d3838035d059178184d1d8f19`.
Las categorías seleccionan páginas; no se atribuye el país de una categoría a
todas las acepciones de la página. Sólo se conserva un sentido si su fuente
explicita al menos un ámbito incluido en la selección regional.

Ejemplos TRAIN comprobados contra su revisión primaria: «guagua» designa un
niño muy pequeño en Chile, entre otros ámbitos; otra acepción designa transporte
colectivo en Cuba, Puerto Rico y República Dominicana, junto con ámbitos ajenos
a Latinoamérica. «Chamba» tiene la acepción trabajo/empleo en México,
Centroamérica y Perú; otra acepción en Colombia describe una zanja o hendidura.
Las etiquetas no afirman que esos sean los únicos usos posibles. Guagua conserva
las notas Argentina/Cuyo y Bolivia/Tarija. Canarias y Guinea Ecuatorial se
conservan como ámbitos originales, sin contarlos como países latinoamericanos.
«Ustedes» aporta aquí una acepción particular de Chiloé; no sustituye la
descripción general del pronombre. Las glosas pueden contener errores de origen
y no se presentan como hechos certificados de forma independiente.

La distribución por país cuenta etiquetas explícitas de acepciones, sin
expandir «Centroamérica», «Rioplatense» o «América» a países supuestos. Una
entrada y una acepción pueden pertenecer a varios países:

| País | Títulos directos catalogados | Entradas TRAIN/VAL con etiqueta | Acepciones TRAIN/VAL con etiqueta |
| --- | ---: | ---: | ---: |
| Argentina | 3.515 | 25 | 36 |
| Bolivia | 844 | 18 | 22 |
| Chile | 3.086 | 38 | 56 |
| Colombia | 1.572 | 31 | 35 |
| Costa Rica | 541 | 13 | 17 |
| Cuba | 930 | 18 | 27 |
| Ecuador | 751 | 13 | 16 |
| El Salvador | 851 | 11 | 12 |
| Guatemala | 486 | 13 | 13 |
| Honduras | 976 | 9 | 13 |
| México | 2.191 | 25 | 37 |
| Nicaragua | 735 | 9 | 9 |
| Panamá | 307 | 6 | 6 |
| Paraguay | 451 | 7 | 7 |
| Perú | 925 | 18 | 22 |
| Puerto Rico | 520 | 11 | 13 |
| República Dominicana | 436 | 15 | 16 |
| Uruguay | 1.714 | 11 | 15 |
| Venezuela | 1.724 | 23 | 38 |

[country-distribution.json](country-distribution.json) también informa títulos
TRAIN, VALIDATION y TEST excluidos del catálogo causal. El
[índice de acepciones](definitions/country-sense-index.json) registra cada ID,
fold, acepción y nota. El [catálogo original parseado](country-catalog.json)
conserva los títulos, IDs, claves de ordenación, timestamps y continuaciones
vacías de cada país. Es inventario de metadatos; el entrenador recibe únicamente
los archivos que terminan en `train-validation.jsonl`.

Las familias conservan los folds anteriores mediante los registros de
Wikcionario existentes y las relaciones explícitas de las entradas adquiridas.
Se agrupan variantes de acento, caja y reflexivos de forma conservadora; no se
inventan relaciones entre todas las flexiones todavía desconocidas. El catálogo
tiene 10.826 familias de inventario: 9.758 con texto TRAIN/VALIDATION y las
restantes con títulos TEST excluidos. **1.111 títulos TEST no se emitieron como
texto causal**. No se movió una familia histórica ni se leyó una suite de
evaluación conversacional para preparar estos datos. Se revisaron todos los
grupos comunes de los dos inputs y ninguno cruzó TRAIN/VALIDATION. Los guards
de glosas largas exactas frente al vocabulario anterior y `wide-spanish-v1`
no hallaron conflictos; no son un detector exhaustivo de paráfrasis.

La API declara **CC BY-SA 4.0** en [siteinfo.json](siteinfo.json), con enlace a
[la licencia](https://creativecommons.org/licenses/by-sa/4.0/deed.es). Los textos
adaptados conservan esa licencia y atribución a los colaboradores de Wikcionario.
Cada glosa fija página, revisión, timestamp, URL `oldid`, historial y hashes del
wikitext y su sección española. El catálogo conserva recibos de las consultas y
enlaces al historial de cada entrada; sus metadatos de pertenencia son una
captura de consulta, no una glosa vinculada a un único `oldid` global.

No se descargó un catálogo propietario de RAE/ASALE. Se excluyeron etimologías,
traducciones, definiciones con referencias, citas acreditadas a libros/canciones,
plantillas desconocidas y ejemplos con metadatos externos. Los cuerpos completos
de revisiones permanecen privados porque pueden incluir esas citas descartadas.
[withheld-source-bodies.json](withheld-source-bodies.json) conserva sus URLs y
hashes. Las respuestas originales que sólo contienen metadatos se distribuyen
comprimidas sin pérdida; [metadata-response-archives.json](metadata-response-archives.json)
registra hashes del original y del archivo, y su vuelta exacta verificada.

La adquisición duró **257,73 segundos**, recibió 2.809.078 bytes y usó pico RSS de
81.000 KiB, CPU1, 512 MiB de espacio de direcciones y límites de 900s/32MiB de
red. Dos HTTP429 se recuperaron con pausas de al menos 30 segundos, manteniendo
un intervalo mínimo de tres segundos entre solicitudes. Quedaron recibos de
ambos. La construcción útil del catálogo duró 3,62s, sin red ni modelos. El
primer intento excedió el límite de 8MiB del corpus por metadatos repetidos;
se conservan fuente y recibo de rechazo. Su salida privada está archivada sin
pérdida. El intento final conserva procedencia en registros y manifiesto y
queda dentro de ese límite. Este lote completo, copias privadas y públicas
incluidas, se mantuvo dentro de 32MiB de disco lógico.

La revisión por agentes de IA del proyecto detectó una omisión de `nota1` en
el registro de uso de «china»: «despectivo (antaño no lo era)» había perdido la
nota. Se reparó el parser después de congelar la primera ingesta, sin volver a
descargar fuentes ni cambiar folds. Se verificaron las 106 revisiones públicas
y se corrigió exactamente ese registro TRAIN. La versión original y los tres
fallos de las pruebas de notas se conservan. **Seis pruebas específicas del
parser pasan** tras normalizar `nota1` y rechazar notas huérfanas o contradictorias.
[El recibo](data-review-and-parser-repair.json) identifica comandos, hashes y
alcance. No hubo revisión humana independiente.

La reproducción portable de acepciones ya se ejecutó sin red ni wikitext
completo y produjo exactamente el SHA256 publicado en 0,011s. Desde la raíz
del repositorio, con una salida nueva:

```bash
python3 examples/rebuild_latin_american_senses.py \
  --parsed-source examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/definitions/latin-american-wiktionary.train-validation.jsonl \
  --expected-source-sha256 d27c633d6166f42d7fe5d59d865b5a8a3ba67c6d3838035d059178184d1d8f19 \
  --output /tmp/unidad-regional-senses-replay

python3 examples/build_latin_american_catalogue.py \
  --catalogue examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/country-catalog.json \
  --expected-catalogue-sha256 520d88ae9a92ca1ace6130dd1a48f103608047d555a7fdfc163511c40f2a95c6 \
  --registry examples/learning-repair-20261013/new-sources/wiktionary-spanish-v1/lexical-family-registry.json \
  --registry examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/definitions/lexical-family-registry.json \
  --output /tmp/unidad-regional-catalogue-replay
```

La segunda receta usa exactamente los mismos metadatos y registros de familias
que la construcción medida; no se ejecutó una segunda copia completa de su
salida para evitar disco redundante. [integration-recipe.json](integration-recipe.json)
identifica ambos inputs por hash para la mezcla posterior. Esta entrega de datos
no encola ni detiene entrenamientos y no demuestra que UNIDAD haya aprendido
estas palabras. Un SFT derivado debe conservar familia, fold y acepción de origen,
pedir contexto cuando haya varias opciones y mantener las notas regionales; no
debe elegir una glosa por país si el mismo país admite varias.
