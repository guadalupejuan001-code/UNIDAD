# Currículo natural y Latinoamérica v4

Corpus final preparado tras revisión de targets, para dos brazos desde el mismo
checkpoint propio: **LATAM primero, control después**, 4.096 actualizaciones y
configuración iguales. Las etapas se encolan desde el hilo principal; esta
preparación no entrenó modelos ni demuestra aceptación conversacional.

| Brazo | TRAIN | VALIDATION | Corpus SHA-256 |
| --- | ---: | ---: | --- |
| Control natural | 9.951 | 1.704 | `48cc11848568c03a32727660968e47695d8eeb43c3302cbf7055eefe2e449491` |
| Natural + LATAM | 10.359 | 1.759 | `cf52daaaa0bfbd6dd539977ebaabb1b4b265d0ad192d9d6cea6c49e18da5515f` |

Ambos mantienen todas las filas humanas y curadas del corpus anterior y sus
targets/folds. Se seleccionaron 400 familias completas de ficción procedural,
diez de cada una de las 40 plantillas TRAIN: 1.598 filas frente a las 10.235
anteriores. Las 1.704 filas VALIDATION originales conservan sus bytes exactos.
La adición LATAM final contiene **463 registros sintéticos del proyecto**:
408 TRAIN y 55 VALIDATION, 89 familias léxicas. Países, significados y notas de
uso proceden de acepciones atribuidas de Wikcionario; sus categorías de página
no se convierten en significados. La muestra menciona los 19 países/territorios
del origen. No contiene todo el español latinoamericano ni mide generalización.

El diseño se fijó antes de preparar cada versión. La revisión de IA del proyecto
rechazó v2 y v3 por referencias sin antecedente, circularidad, gramática y datos
factuales dudosos de la fuente. V4 elimina las seis filas restantes de caraota
y algarrobilla y conserva **literalmente todos los demás registros de v3**.
Por eso sus etiquetas internas siguen diciendo `v3`: identifican la generación
de origen, no una regeneración. El
[recibo final](final-review-exclusions.json) y las
[exclusiones previas](preceding-target-exclusions.json) conservan las decisiones
y los textos originales. Los conjuntos anteriores permanecen como evidencia;
no se reescribieron hechos biológicos para convertirlos en targets.

Se conserva `assistant_last_reply`. El asistente incorrecto de un historial
sintético es contexto para recuperarse, nunca un objetivo positivo. Se entrenan
significados por país, correcciones del país, interlocutores, formato solicitado
y aclaración ante ambigüedad real. Hay ocho patrones programáticos; sus variantes
no se contabilizan como hechos nuevos. La revisión de todos los targets y nueve
pruebas de contratos de datos, incluidas seis referencias de acepción/país, son
comprobaciones del proyecto mediante IA y código, no evaluación humana.

La preparación v4 tardó 0,472 segundos y alcanzó 150,5 MiB de RSS. No cargó
modelos/tokenizadores ni leyó TEST. Las copias locales inmutables comparten inode
para conservar las tres versiones dentro de 64 MiB físicos; Git conserva sus
blobs normales. Nunca se modifica un corpus existente. La longitud con el
tokenizador propio debe comprobarse antes de admitir la etapa.

La comparación usa los mismos casos externos de desarrollo, el mismo motor,
historia realmente generada y los mismos límites. Las VALIDATION de los brazos
son diferentes: sus NLL globales no son una comparación equivalente. El criterio
final sigue siendo 36/40 respuestas completas, 8/10 diálogos totalmente correctos
y cero fallos críticos/técnicos en una prueba nueva tras fijar candidato y
configuración. Preparar datos o bajar la pérdida no cumple ese criterio.

Reproducción desde v3 congelada, en una carpeta nueva:

```bash
python examples/finalize_natural_latam_sft.py \
  --previous-v3 examples/learning-repair-20261013/natural-latam-sft-v3 \
  --design examples/learning-repair-20261013/natural-latam-sft-v4/prospective-design.json \
  --output-dir private/reproduction-natural-latam-v4
```

[Control](natural-control.manifest.json) y
[LATAM](natural-latam.manifest.json) incluyen hashes del corpus y del contenido
conversacional sin metadatos. La [receta](recipe.json), el helper congelado y
`qa` permiten revisar la preparación. Los tiempos medidos no son una promesa de
rendimiento en otra máquina.

Las adaptaciones regionales conservan CC BY-SA 4.0, autores mediante historial,
revisión y URL de licencia por registro. La colección preserva las condiciones
por componente del [corpus SFT anterior](../LICENSE-DATA.md); OpenAssistant mantiene
Apache 2.0. Agrupar registros no cambia sus licencias originales.
