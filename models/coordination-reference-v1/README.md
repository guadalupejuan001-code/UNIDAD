# Selector de coordinación entrenado v1

Selector lineal propio con 18 pesos, inicializados aleatoriamente y entrenados
para preferir candidatos reales que cumplen criterios literales acotados. No
genera texto: elige una respuesta original del conjunto de Transformers.

Identidad: `unidad-selector-9a2eefa31085a181bc488bf50455ea2fff2d8feee3f0961a95d07b644c4b3154`.
SHA-256 del archivo `model.json`:
`88068c978e5aea5531af3fb1400466a507a97b43b706a4e54bccf20854fd2ff6`.

Completó 1.000 épocas sobre 38 preferencias TRAIN y 15 VALIDATION, con familias
ficticias separadas. El entrenamiento de los pesos duró 0,107 segundos; generar
los datos con los Transformers requirió unos 51 segundos. Los pesos se
congelaron antes de abrir los 12 casos TEST.

En TEST obtuvo 2/12 cumplimientos literales, igual que las reglas anteriores.
Hubo una ganancia y una regresión; 7/12 conjuntos no contenían ningún candidato
que cumpliera el criterio. No se promovió como selección predeterminada. Está
conectado mediante `configs/sovengi-coordination-candidate.json` para ejecución
experimental explícita.

Una revisión cualitativa de IA encontró cero respuestas completas en ambos
brazos. La coincidencia literal con el dato no acredita una respuesta natural.

El [informe](../../docs/sovengi-coordination-20261013/README.md) conserva datos,
pesos, comparación, respuestas originales y límites. Sus 18 características
no son una validación semántica ni una capacidad conversacional aceptada.
