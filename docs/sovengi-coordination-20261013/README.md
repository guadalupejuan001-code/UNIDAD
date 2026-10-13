# Entrenamiento de la coordinación de SOVENGI 0.9

Esta etapa prioriza entrenar la conexión que elige entre respuestas mientras
la publicación de la integración anterior continuaba en paralelo. Esa entrega
está fusionada mediante el [PR 21](https://github.com/guadalupejuan001-code/UNIDAD/pull/21).

## Pesos y ejecución real

Se entrenó un selector lineal de 18 parámetros desde pesos aleatorios propios.
Recibe preferencias relativas de los críticos, apoyo léxico, coincidencia con
la pregunta, diagnóstico RNN, longitud y repetición. Los identificadores de los
Transformers no son características del aprendizaje. La salida sigue siendo
exactamente el texto y los tokens de un candidato original.

La etapa recolectó 36 casos TRAIN y 12 VALIDATION con los 35 Transformers reales,
cuatro candidatos por petición, dos críticos y hasta cuatro motores residentes.
Todos los datos y nombres son ficticios nuevos. Los 12 casos TEST se almacenaron
en una partición separada y se abrieron después de congelar los pesos.

El objetivo de entrenamiento es la pérdida logística de preferencias entre
candidatos con utilidades distintas. Los empates se excluyen. Se completaron
1.000 épocas, con 38 pares TRAIN y 15 VALIDATION. La validación seleccionó la
época 1.000. La pérdida TRAIN pasó de 0,69994 a 0,04589; esta pérdida no mide
calidad conversacional. El ajuste duró 0,107 s y la recolección unos 51 s.

## Comparación reservada

Cada comparación usa el mismo conjunto realmente generado. Las utilidades
comprueban palabras de una ubicación, el día corregido sin el anterior o una
expresión explícita de desconocimiento. Son criterios literales conservadores;
pueden omitir paráfrasis válidas y aceptar texto deficiente que contenga el dato.

| Partición | Reglas anteriores | Selector entrenado | Conjunto con algún candidato que cumple |
| --- | ---: | ---: | ---: |
| TRAIN | 9/36 | 12/36 | 12/36 |
| VALIDATION | 4/12 | 4/12 | 5/12 |
| TEST reservado | 2/12 | 2/12 | 5/12 |

TEST se ejecutó una vez con los pesos congelados y el selector aplicado en las
12 peticiones reales de SOVENGI. No hubo errores técnicos. Ganó una ubicación
y perdió una corrección; no consiguió mejora neta. En siete casos ningún
candidato cumplía el criterio, de modo que la selección no podía resolverlos.
Se conserva como candidato conectado y la configuración habitual mantiene las
reglas anteriores. Esta decisión no modifica pesos ni reutiliza TEST para otro
ajuste.

La revisión cualitativa posterior de un agente de IA encontró **0/12 respuestas
completas en ambas selecciones**: las reglas tuvieron tres parciales y nueve
fallidas; el selector, cuatro parciales y ocho fallidas. Algunos candidatos
contenían el día esperado dentro de frases incoherentes. La revisión se conserva
como inspección de IA, sin evaluación humana independiente ni aceptación final.

## Uso y evidencia

```bash
python -m pip install -e '.[conversation]'
sovengi --config configs/sovengi-coordination-candidate.json models
sovengi --config configs/sovengi-coordination-candidate.json ask 'Hola'
```

La traza `ensemble.learnedSelection` identifica los pesos usados, sus scores y
el candidato elegido; `ruleSelectedModelId` conserva la comparación anterior.
Un selector inválido o sin presupuesto deja un motivo explícito y usa la regla
anterior. La configuración y la ruta del artefacto pertenecen al anfitrión.

- [Artefacto entrenado](../../models/coordination-reference-v1/model.json).
- [Corpus y protocolo fijados](../../examples/sovengi-coordination-20261013/corpus.json).
- [Recolección original](collection.json) y [predicciones de desarrollo](development-predictions.json).
- [Congelación previa a TEST](selector-freeze.json).
- [Informe completo y respuestas reservadas](report.json).
- [Revisión cualitativa de IA](semantic-review.json), separada del criterio literal.
- [Programa de entrenamiento](../../examples/train_sovengi_coordination.py).

El programa separa `collect`, `fit` y `evaluate` y rechaza regenerar una
recolección o repetir un test ya cerrado. Los límites son 180 segundos para
recolección, 120 para ajuste y 180 para evaluación, comprobados entre
operaciones. Los recibos registran hashes de código por fase: la recolección
conserva su versión original y la evaluación usa la integración final del
selector. El commit base y esos hashes identifican el código realmente usado.

Los pesos de los Transformers y de la RNN se usaron para generar y puntuar;
el entrenamiento de esta etapa ajustó los 18 pesos del coordinador. No se
usaron conversaciones personales, servicios de modelos externos ni ajustes
posteriores a la evaluación reservada. No hay un entrenamiento continuo activo
prometido por estos artefactos.
