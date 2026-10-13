# Español literario histórico: selección Gutenberg v2

Este paquete contiene **177 fragmentos, 482.039 palabras y 2.904.931 bytes de
JSONL TRAIN+VAL**, preparados para una etapa causal posterior. No se ha entrenado
un modelo con este paquete al prepararlo. No representa todo el español ni una
receta de conversación contemporánea.

| Autor y obra | Partición | Fragmentos | Palabras |
| --- | --- | ---: | ---: |
| Cervantes, *Don Quijote*; Quevedo, *Buscón*; anónimo, *Lazarillo* | TRAIN | 161 | 440.178 |
| Fray Luis de León, *La perfecta casada* | Validación de desarrollo | 16 | 41.861 |

Las obras y todos sus fragmentos conservan la familia completa del autor:
`public-spanish-author:<slug>`, con `assign_split` del proyecto. La misma clave
se comparte con los otros importadores públicos. La validación contiene un solo
autor histórico y no mide conversación moderna. El texto contiene ortografía,
estilo y opiniones de su época; debe mezclarse con anclas modernas y acompañarse
de SFT y evaluación de respuestas reales.

La reserva de corpus contiene 21 fragmentos de Calderón, con 53.182 palabras,
separados y sin tokenizar. No se distribuye aquí ni se incluye en TRAIN+VAL. Su
prefacio crítico de Menéndez y Pelayo y el catálogo editorial se retiraron para
que la familia corresponda al dramaturgo. La edición leída es de **1881**.

## Procedencia y condiciones

Las fichas y textos se consultaron directamente en Project Gutenberg el
13 de octubre de 2026 UTC: [Don Quijote, #2000](https://www.gutenberg.org/ebooks/2000),
[Buscón, #32315](https://www.gutenberg.org/ebooks/32315),
[Lazarillo, #320](https://www.gutenberg.org/ebooks/320),
[La perfecta casada, #61573](https://www.gutenberg.org/ebooks/61573) y
[Calderón, #54436](https://www.gutenberg.org/ebooks/54436).
Las fichas declaran dominio público en Estados Unidos; los autores originales
seleccionados fallecieron en los siglos XVI y XVII, salvo el anónimo de 1554.
Esa declaración estadounidense no se presenta como una conclusión universal
sobre todas las ediciones ni los territorios. Se conservan textos originales,
la [licencia de Gutenberg](https://www.gutenberg.org/policy/license.html),
su [explicación de derechos](https://www.gutenberg.org/help/copyright.html),
fichas, URL, fecha, encoding UTF-8, cabeceras y SHA-256 en `raw/` y `provenance/`.
Las fichas incluyen resúmenes automáticos del sitio: **son documentación de
procedencia y no se usan como texto de entrenamiento**.

La primera preparación privada incluía *Fuente Ovejuna* #60198 en TEST. La
inspección del propio texto identificó la edición de Américo Castro, de 1919.
Se excluyó esa edición entera en v2 por ese colaborador reciente; su contenido
y texto bruto no se publican en este paquete. También se rechazó *Tirso de
Molina* #58194, por su editor reciente. Las decisiones quedan en `catalog.json`;
los preparativos v1 permanecen como evidencia privada. Ninguna de esas obras
estuvo en TRAIN+VAL, cuyo hash permaneció igual al pasar a v2.

## Archivos y reproducción

El hash de `gutenberg-spanish.train-validation.jsonl` es
`96c58c71a5834c3503fe4ed69b25b315ff1cf0b37414c571d7ab5623e66f4ce3`.
`manifest.json` incluye los campos usados por el supervisor: `corpusSha256`,
`splitCounts` con TEST cero, `testUsedForTraining: false` y `sources`.
`preparation-manifest.json` conserva el resultado completo de la preparación
privada; `distribution-receipt.json` fija los archivos distribuidos antes de
añadir esta explicación. La copia congelada del helper permite auditar el
código que preparó esos datos.

Con el proyecto instalado, la reproducción completa descarga solamente las
cinco fuentes fijadas, comprueba sus hashes y vuelve a reservar Calderón:

```bash
python examples/prepare_gutenberg_spanish.py \
  --catalog examples/learning-repair-20261013/new-sources/gutenberg-classic-es-v2/catalog.json \
  --output-dir private/gutenberg-spanish-reproduction \
  --max-seconds 600 --max-rss-mib 512 --max-disk-mib 256 \
  --max-network-mib 64 --max-corpus-mib 10
```

Una actualización del archivo remoto que cambie el hash provoca un rechazo.
No se cambia silenciosamente la fuente. Los cuatro textos brutos de TRAIN+VAL
también se distribuyen congelados. Se retira el encabezado y pie Gutenberg,
el prefacio moderno de transcripción y párrafos cortos, de control o ajenos
al texto; los fragmentos conservan las palabras originales.

La descarga inicial de las seis fuentes empleó 3.570.404 bytes y 7,42 segundos.
La preparación final v2 desde textos locales consumió 0,587 segundos y un pico
de 37,277 MiB según el registro nativo. Se fijaron un núcleo, 512 MiB de espacio
de direcciones, 256 MiB de disco y 600 segundos; no hubo entrenamiento auxiliar,
señales al entrenador ni batería adicional de software. Se conservó un fallo
inicial: dos notas de transcripción idénticas generaban una colisión entre
autores de VAL y TEST. Se retiró el prefacio completo antes de deduplicar, sin
mover autores entre particiones ni usar resultados de evaluación.

La preparación no creó una cola independiente. El responsable del ciclo puede
combinar estos textos con fuentes contemporáneas identificadas y ajustar después
la conversación; la corrida existente continúa con sus archivos fijados.
