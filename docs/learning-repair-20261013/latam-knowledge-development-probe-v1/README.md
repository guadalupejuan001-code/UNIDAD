# Prueba diagnóstica de aprendizaje LATAM

Quince preguntas de desarrollo congeladas antes de generar respuestas: seis
paráfrasis y seis combinaciones de seis hechos TRAIN, más tres preguntas VAL
que admiten incertidumbre. Los seis países TRAIN son distintos. Las dos preguntas
de cada lexema comparten familia y hecho; miden generalización local de pregunta
y formato, sin acreditar generalización a lexemas inéditos.

[Casos](cases.json), [rúbrica](rubric.json), [preregistro](preregistration.json),
[fuentes con bytes exactos](source-support.json) y [auditoría de separación](novelty-audit.json)
se identifican mediante [freeze.json](freeze.json). La revisión de fuentes es de
IA del proyecto, sin validación humana independiente. Las adaptaciones conservan
CC-BY-SA-4.0, atribución e historial/revisión de Wikcionario por registro.

Las familias y lexemas VAL elegidos están ausentes del TRAIN v4 actual; la base
puede haberlos encontrado en corpus anteriores. Sus tres hechos no se exigen
como aprendizaje: registrar respuesta correcta, incertidumbre explícita o error
por separado. Los nombres nuevos y preguntas no aparecen en TRAIN v4; ninguna
glosa esperada se entrega al modelo.

La comparación fijada usa el padre 072, BEST ec9, LAST d6 y el control LAST cuando
cierre, con el mismo `ConversationEngine`, temperatura 0, máximo 80 tokens,
semilla 42 y un hilo CPU. El ID aún desconocido del control se fija en un recibo
separado antes de generar, siguiendo el endpoint procedural predefinido. Este
paquete no genera respuestas ni entrena modelos. El runner de 40 turnos existente
no es compatible con 15 preguntas independientes y máximo 80 tokens.

No usar estos casos ni resultados para reentrenar o elevar sus propios recuentos.
No acreditan aceptación final, conversación completa ni todo el español. La
aceptación reservada permanece separada.

La [preparación](prepare.frozen.py) usa únicamente la biblioteca estándar y
corpus públicos congelados. Escribe archivos nuevos una sola vez y comprueba
hashes, particiones, país/sentido, exclusiones y ausencia literal de prompts y
nombres antes de guardarlos. No importa modelos ni lee la evaluación reservada.
