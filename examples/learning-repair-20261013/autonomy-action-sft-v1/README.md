# Supervisión de acciones con archivos reales

192 TRAIN y 24 VALIDATION de 72 tareas ficticias separadas por tarea, familia, nombres y plantillas de objetivo. Cada etiqueta JSON fue escrita por una fixture declarada del proyecto. El controlador y SafeFileTools ejecutaron 216 acciones sobre archivos temporales reales, verificaron informes y conservaron originales por SHA-256; las raíces se destruyeron después del recibo.

No se cargaron pesos ni se generaron respuestas con modelos. Estos datos no demuestran planificación aprendida ni autonomía fiable. El experimento futuro registrado propone un control pequeño de memorización y luego DEV con tareas y nombres nuevos, usando pesos propios y contexto 2048. La preparación no entrena ni encola etapas.

[Manifiesto](manifest.json), [diseño previo](preregister.json), [QA](qa.json), [recibos reales](execution-receipts.jsonl), [licencia](LICENSE-DATA.md). La auditoría usa el BPE2048 propio y ambos codificadores de ventanas nativos congelados; supervisa sólo el último JSON y su EOS, con todo el contexto y target dentro de 2048. TEST no se leyó ni se incluyó.

Reproducción desde la raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python3 examples/prepare_autonomy_action_sft.py --output-dir /tmp/unidad-action-sft-new`. Cada ejecución escribe una carpeta nueva y no sustituye evidencia anterior.

La primera preparación con 1024 se bloqueó en la tercera propuesta. El [diseño previo conservado](failed-generic1024-preregister.json) y las [mediciones reproducidas](failed-generic1024-measurements.json) registran 816/1064/1298 tokens de entrada de entrenamiento y un último prompt de 1201; 1024 descartaría 177 tokens y el adaptador bloquea antes de generar. La reproducción diagnóstica añadió cero registros al corpus. Los 72 controles negativos del corpus invocaron realmente el adaptador de 1024 y comprobaron cero llamadas de generación.
