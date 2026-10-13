# Collector Wikipedia V2: reparación de admisión de lotes

El collector original abortó el 13 de octubre después de descargar 69 páginas.
Su comprobación mezclaba artículos nuevos con cinco clases de párrafos que ya
existían entre TRAIN y validation del rehearsal congelado. El constructor inicial
comparaba mensajes OASST completos; el collector introdujo después una comparación
de párrafos, incluidos títulos breves y un párrafo vacío. El fallo no acreditaba
una colisión nueva de Wikipedia. La [auditoría](collector-v2-collision-audit.json)
conserva hashes, familias y procedencia. El script original mantiene SHA-256
`feee09834049f1f163e7e98badcb2dfaf9a0c00f2bde65cef6191ef1199619c9`.

La versión nueva está en `examples/collect_learning_wiki_v2.py`. Conserva todas
las filas y folds del rehearsal; declara sus clases históricas por separado.
Rechaza familias nuevas completas que compartan párrafos entre folds, incluso
cuando ambos participantes son nuevos. Un duplicado exacto excluye la familia
nueva; una coincidencia parcial dentro del mismo fold conserva el contenido
nuevo restante. Identidades de familias fijas nunca se reutilizan ni se trasladan.
El historial completo del seed permanece sin cambios; TEST sólo aporta las
identidades y digests previamente registrados, y nunca texto decodificado.

Cada respuesta nueva conserva raw y recibo con revisión del snapshot, SHA-256,
offset, orden y URL. Un intento fallido no avanza el cursor ni publica aprobación;
el siguiente intento puede reutilizar esos mismos bytes. No se recuperaron las
69 respuestas antiguas: el original las eliminó al limpiar staging. Un cache
diagnóstico anterior sin recibo no se usó como ingesta acreditada.

La [reproducción real](collector-v2-preparation-evidence.json) preparó tres páginas
desde el cursor 593: reutilizó la primera respuesta tras un intento que carecía de
nueva validation y obtuvo dos páginas adicionales, con una petición por página
y límite de 15 segundos por petición. El presupuesto total fue 60 segundos.
El lote contiene 109 registros Wikipedia nuevos y conserva los 2.299 anchors;
no incluye TEST ni mueve folds. Quedó `prepared_not_admitted`: no se escribió
state del collector ni aprobación o bundle en inbox. La retención SFT del preview
usa el corpus balanceado V4 y su configuración de 4.096 pasos, ambos fijados por
hash. No se midió mejora del modelo.

La [receta de preview](collector-v2-preview-recipe.json) es reproducible con:

```bash
python examples/collect_learning_wiki_v2.py collect \
  --recipe docs/learning-repair-20261013/collector-v2-preview-recipe.json \
  --inbox-dir private/learning-repair-20261013/collector-v2-preview/inbox \
  --supervisor-state private/learning-repair-20261013/continual/state.json \
  --prepare-only --preview-parent-stage conversation-v1:conversation-15m
```

Una preparación existente se conserva y no se sobrescribe. Para otra reproducción
se elige un outputRoot nuevo; pueden copiarse los raws y sus recibos verificados
al subdirectorio `raw-cache-v2` de ese outputRoot y fijar `offline: true` en una
receta nueva. La instalación futura requiere una receta y configuración nuevas
con sus propios source pins; esta reparación no modifica la configuración activa,
el collector original, presupuestos cerrados ni fuentes congeladas del entrenador.

Siete regresiones comprueban el baseline real, exclusión completa de familias,
coincidencias seguras dentro de un fold, duplicados exactos, raw cache verificado,
TEST sin decodificar y fallos de tiempo/disco sin publicación o avance de cursor.
