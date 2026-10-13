# Copia de argumentos de archivos v2

192 TRAIN y 24 VALIDATION de 64/8 tareas completas. Las etiquetas son JSON correcto de una fixture declarada; SafeFileTools y el controlador ejecutaron 216 acciones reales, verificaron informes y conservaron originales por hash. Se reutilizó el ejecutor v1 con sus tres herramientas y esquemas genéricos sin cambios.

La intervención cambia sólo la distribución de datos para copiar nombres de archivo y de informe del objetivo y task_inputs actuales: nombres de una a tres palabras, tildes, espacios, guiones y extensiones txt/md/csv; orden del objetivo y de la lista de entradas variable. Los nombres, familias y plantillas de objetivo de VALIDATION quedan separados de TRAIN; todos son ficción, sin nombres del demo observado. Los datos v1 permanecen intactos. No se enseñó como correcto el nombre rechazado que produjo el modelo.

[Diseño previo](preregister.json), [manifiesto](manifest.json), [QA y estadísticas BPE](qa.json), [exclusiones y splits](needle-exclusion-and-split-audit.json), [recibos reales](execution-receipts.jsonl), [recetas futuras](future-matched-recipes.json), [licencia](LICENSE-DATA.md). La auditoría nativa supervisa únicamente el último JSON y su EOS; cada prompt completo más target cabe en 2048. TEST no se leyó.

El contraste propuesto continúa ambos brazos desde el mismo E202 propio: v2 frente a ensayo v1, configuración idéntica, máximo 256 actualizaciones o 600 segundos nativos y un solo entrenador. Esta preparación no entrenó, no encoló y no cambió arquitectura, pérdida ni generación. Validez JSON y NLL bajas no demuestran argumentos autorizados, finalización real, generalización ni competencia conversacional en español.

Reproducción desde la raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python3 examples/prepare_autonomy_action_copy_sft.py --output-dir /tmp/unidad-action-copy-new`. Escribe una carpeta nueva; nunca sustituye v1.
