# Currículo pequeño de patrones conversacionales

48 TRAIN y 24 VALIDATION, en 12/6 familias completas con dos contextos contrastantes y dos respuestas correctas por rama. Son ejemplos ficticios redactados por un agente de ingeniería de IA; no son datos humanos ni un catálogo del español. Se preparan para comprobar aprendizaje contextual, sin cambiar la inferencia ni responder con plantillas en ejecución.

Cubre atribución de hechos a personas, correcciones, conocimiento ausente, negación, formato de la petición vigente e información relevante entre distractores. La validación usa nombres, valores, familias y formulaciones separados y combina seguimiento de hechos con formatos. No constituye aceptación final ni acredita generalización amplia.

El entrenador nativo etiqueta únicamente la última respuesta correcta y EOS. Los asistentes erróneos de la familia de correcciones están exclusivamente en historia y todos sus labels son ignorados; cada prefijo positivo termina en una respuesta corregida. El manifiesto y las auditorías conservan las entradas y targets completos.

`development-cases.json` separa contexto inicial, turnos de usuario y respuestas esperadas. En historia generada se añaden respuestas reales del modelo; los targets nunca se inyectan. La historia ideal de los prefijos es sólo un diagnóstico por separado.

Preparación desde la raíz: `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 examples/prepare_conversation_pattern_curriculum.py --parent RUTA_MODELO_PROPIO --output-dir RUTA_NUEVA`. No entrena, no encola, no toca datos anteriores y rechaza sobrescribir una entrega.
