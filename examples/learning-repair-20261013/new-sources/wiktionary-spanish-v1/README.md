# Vocabulario y formas españolas de Wikcionario

Ingesta original realizada el 13 de octubre de 2026 con la API pública de
[Wikcionario](https://es.wiktionary.org/w/api.php). El corpus causal TEXT publicado
contiene **614 entradas TRAIN y 105 VALIDATION**, 1.065.856 bytes, SHA256
`8dd844e98d0a2d1147f962cbe7ed2eeb54fecb0d1b906d8b22a4bfbca531c519`.
Sus textos contienen 33.775 palabras TRAIN y 4.685 VALIDATION, 2.022 definiciones
y 681 ejemplos de uso entre ambos folds. TRAIN contiene 5.565 formas léxicas
distintas contando palabras alfabéticas sin distinguir mayúsculas. Esa cifra no
es el tamaño del vocabulario del tokenizador ni demuestra conocimiento aprendido.

Se consultaron 1.200 páginas, se recibieron 1.199 revisiones y se conservaron
804 entradas útiles de 705 familias,
con artículos, pronombres, preposiciones, conjunciones, interjecciones, locuciones,
sustantivos, adjetivos, verbos y adverbios. La selección usa palabras funcionales
explícitas y once anclas alfabéticas por categoría amplia. Las revisiones se
descubrieron en la API y después quedaron fijadas mediante IDs, URL `oldid`, fecha,
hash del wikitext y hash de la sección española. Es una muestra acotada y sesgada
por esas categorías y anclas; no contiene todo Wikcionario ni todo el español.

La API original `siteinfo.rightsinfo` declara **CC BY-SA 4.0**, con enlace a
[la licencia](https://creativecommons.org/licenses/by-sa/4.0/deed.es).
Cada entrada conserva título, página, revisión, URL del historial y atribución a
los colaboradores de Wikcionario. La adaptación limpia distribuida mantiene esa
licencia y atribución; no hereda automáticamente la licencia del código de Unidad.
La declaración general del sitio no certifica por separado los derechos de cada
obra subyacente citada. El extractor excluye definiciones con referencias y
ejemplos acreditados de libros, canciones, webs o diccionarios modernos; no se
descargó el corpus propietario de la RAE para entrenar.

`examples/prepare_spanish_wiktionary.py` conserva únicamente la sección de lengua
española, glosas y ejemplos sin cita externa. Elimina enlaces y marcado, rechaza
templates de contenido desconocidos y representa un conjunto pequeño de templates
explícitos de flexión y comparativo mediante texto español. No inventa paradigmas
completos: las tablas conjugadas por plantillas, traducciones y etimologías quedan
fuera. La fuente puede contener errores, ortografía histórica o imprecisiones;
no es una base de hechos verificada. La revisión de doce entradas TRAIN fue hecha
por un agente de IA del proyecto, sin revisión humana independiente. Se conserva
un ejemplo de posible errata de origen en el recibo editorial, sin reescribirla
silenciosamente.

Las familias son componentes de relaciones explícitas entre lemma, formas y
variantes ortográficas. Se normalizan acentos, mayúsculas y variantes reflexivas
de forma conservadora. Los códigos de persona, tiempo y género no se usan como
raíces compartidas. `assign_split` asigna el componente entero a TRAIN, VALIDATION
o TEST. `lexical-family-registry.json` registra los alias y folds que deben
respetarse en futuras ampliaciones; un conflicto entre folds anteriores debe
quedar en cuarentena, sin mover una familia histórica. Los 85 registros TEST
permanecen en un archivo separado fuera del entrenamiento; no se ajustó ni utilizó
un tokenizador con ellos. La fuente parseada pública permite reconstruirlos,
pero sólo el archivo que termina en `train-validation.jsonl` es entrada de
entrenamiento.

La mezcla amplia `wide-spanish-v1` incorporó **611 TRAIN y 101 VALIDATION** de
esta fuente. Su guard de solapamiento excluyó siete familias completas por
fragmentos de doce palabras o más repetidos entre folds: `diablo`, `diantre` y
cinco variantes de «a la tercera… vencida». No se cambiaron IDs, folds, texto ni
atribución del original. El manifiesto de la mezcla conserva las razones; el
corpus original y su registro de 705 familias permanecen como evidencia.

La adquisición útil duró **374,99 segundos**, con 2.679.846 bytes de respuestas,
CPU1 y límite de espacio de direcciones de 512 MiB. La API devolvió HTTP429 y la
ampliación se cerró con lo realmente adquirido; se respetaron límites de 400s,
30MiB de respuestas, 4MiB de corpus y 128MiB de disco. Dos intentos anteriores
no llegaron a descargar contenido léxico: sus recibos y listados se conservan
en `failed-attempts/`. Los cuerpos originales completos de las revisiones quedan
privados porque pueden incluir citas subyacentes descartadas. Sus URLs y hashes
están en el manifiesto y en `withheld-original-response-bodies.json`. Los recibos
de licencia, metadatos y listados sí se distribuyen. Ningún cuerpo omitido es
necesario para entrenar con el corpus limpio o reconstruirlo.

La reproducción portable usa `acquired-pages.jsonl`, que es una fuente **parseada
y limpia**, no wikitext original. Se ejecutó sin red, modelos ni tokenizador y
reprodujo exactamente los hashes y tamaños de TRAIN/VALIDATION y TEST en 0,063s,
con pico RSS de 38.172KiB. Desde la raíz del repositorio:

```bash
python3 examples/replay_spanish_wiktionary.py \
  --source-dir examples/learning-repair-20261013/new-sources/wiktionary-spanish-v1 \
  --expected-manifest-sha256 610a84a5c45ddb57043f13fff564ebd54d54dc2d9d1a91ec84fa8703bb96c38b \
  --expected-parsed-sha256 a291d56201f6b1be56a4184dd18c349573c2ae9aa08ab779bc20ea0512d4c36c \
  --output /tmp/unidad-wikcionario-replay
```

La salida debe ser nueva. `portable-replay-receipt.json` documenta la ejecución
real y `package-receipt.json` identifica los componentes. Ejecutar la adquisición
en línea descubre otras revisiones actuales y produce otra ingesta: debe recibir
sus propios hashes y conservar el registro de familias antes de integrarse.
Estos archivos amplían datos; la incorporación a una corrida y la capacidad de
conversar sólo se atribuyen cuando existan pesos y respuestas reales comprobados.
