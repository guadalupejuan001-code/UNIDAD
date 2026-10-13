# Currículo natural/LATAM v2: conservado, rechazado para entrenamiento

Estos datos se prepararon antes de entrenar, pero la revisión de IA del proyecto
encontró targets defectuosos aunque copiaban fielmente la fuente: referencias sin
antecedente, definiciones circulares, erratas y datos biológicos dudosos. La
[versión corregida v3](../natural-latam-sft-v3/README.md) excluye esas acepciones.
V2 no se admite como corpus positivo. Sus archivos y el helper congelado se
conservan para que la corrección no borre la evidencia.

Control: 9.951 TRAIN y las 1.704 VALIDATION originales; tratamiento: 10.397 TRAIN
y 1.759 VALIDATION. La adición regional provisional tiene 501 filas. El corpus
original, las particiones y los targets humanos/curados no se cambiaron. No se
cargaron modelos, tokenizadores ni TEST durante esta preparación.

Los hashes completos están en [manifest.json](manifest.json). La procedencia,
acepciones y licencias originales quedan en cada registro. El
[diseño previo](prospective-design.json) y las mediciones se conservan; no son
resultados de un modelo entrenado.
