# Recolección incremental de texto español

`examples/collect_learning_wiki.py collect` produce un lote finito de texto nuevo y una pareja de etapas propias causal → SFT. Lee exclusivamente la instantánea autorizada `wikimedia/wikipedia`, `20231101.es`, revisión `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Reutiliza el modelo y tokenizer del último SFT terminado por el supervisor; no carga pesos durante la recolección ni vuelve a ajustar el vocabulario. El corpus conversacional exacto se reutiliza como retención, no como diversidad nueva.

`recipe.json` usa rutas relativas al repositorio. Los destinos en `private/` se crean al ejecutar la receta y no son dependencias privadas necesarias para clonar el proyecto. El estado del supervisor se suministra por argumento; el supervisor sustituye únicamente `{inbox_dir}` en la lista argv. No se usa un intérprete de shell.

```sh
python examples/collect_learning_wiki.py collect \
  --recipe examples/learning-repair-20261013/collector/recipe.json \
  --inbox-dir private/learning-repair-20261013/loop-inbox \
  --supervisor-state private/learning-repair-20261013/continual/state.json
```

La ejecución requiere el estado `waiting_for_data`, sin worker activo y con un SFT propio terminado. Si todavía está entrenando, informa `waiting_for_completed_sft` sin descargar. La fuente se recorre mediante `offset=((order*7919+37) mod ceil(1841155/100))*100`, sin regresar al comienzo de la permutación. La semilla fija ya vistos 5.200 IDs de las 52 páginas iniciales, incluidos artículos asignados a TEST o rechazados, y empieza en la orden **593**. Los campos TEXT reservados se omiten antes de decodificar JSON; se usan solamente IDs y hashes de metadatos para la semilla. Las páginas nuevas de TEST conservan su ID reservado y su texto no se decodifica.

Cada lote tiene un máximo de 24 MiB de JSONL final, 80 páginas, 900 segundos, 512 MiB de respuestas de fuente aceptadas y 2 GiB acumulados para lotes, estado y bandeja. Las solicitudes usan el proxy y la validación TLS normales, con respuestas individuales de 25 MB como máximo y reintentos finitos. No se garantiza que un lote alcance la cantidad deseada antes de agotar la fuente o los límites. Se exige texto nuevo en TRAIN y VALIDATION para publicar.

La semilla contiene hashes de artículos/prosa completa de todos los folds y hashes de párrafos/chunks de TRAIN y VALIDATION. Se conservan particiones por ID; los IDs ya vistos y los digests conocidos no se reutilizan como contenido nuevo. Se excluyen familias completas con párrafos o chunks repetidos entre TRAIN y VALIDATION; los duplicados exactos posteriores del mismo fold también excluyen la familia nueva completa. Como no se decodifica el TEST histórico, el aislamiento exacto de cada párrafo/chunk reservado frente a otro artículo nuevo no se certifica exhaustivamente. Los hashes no detectan paráfrasis semánticas.

`oasst-rehearsal.train-validation.jsonl` fija 2.059 mensajes de TRAIN y 240 de VALIDATION en familias completas, extraídos del corpus causal inicial por orden de hash de familia y un tope de 3 MiB. Mantiene los textos/folds originales y Apache-2.0; se identifica como retención. Sus 2.299 registros no son fuentes nuevas.

La salida estándar contiene JSON por página y un resultado final `queued`, `publication_recovered`, `waiting_for_completed_sft` o `previous_collector_cycle_pending`. Una publicación completa escribe corpus, manifiesto, recibos de páginas, procedencia de SFT de retención y journal; luego publica `wiki-cycle-NNNN.json` y, al final, su aprobación en la bandeja. El journal y estado persistente permiten completar la misma transacción después de una interrupción. El padre se obtiene de `stageResults.kind=sft` por orden de finalización, sin usar NLL o casos de evaluación. SFT tiene `parentStage=wiki-cycle-NNNN:base`. La aprobación registra la autorización permanente del propietario y `independentHumanReview=false`; no certifica conversación ni promueve el modelo.

La receta y la implementación se prepararon sin ejecutar otra descarga ni pruebas de software durante el entrenamiento activo. El recolector se activa por recarga del supervisor cuando éste espera datos. La semilla y las anclas sí se produjeron mediante operaciones de metadatos/IO sobre las fuentes iniciales ya fijadas. Las licencias están en `../LICENSE-DATA.md` y `../licenses/`; el snapshot de 2023 no garantiza noticias actuales.
