# Generación: proyectar sólo la posición necesaria

La generación calculaba una distribución de vocabulario para cada posición del
contexto y conservaba únicamente la última. Ahora usa `selected_logits` sobre
esa última posición. Todos los tokens siguen pasando por los bloques del
Transformer y siguen disponibles para atención y copia; no se acorta el contexto.

No se modifican pesos, tokenizador, temperatura, semilla, presupuesto de salida
ni selección de tokens. Esta corrección de coste no acredita mayor calidad
conversacional ni la reducción solicitada de fallos semánticos.

## Medición

Mismos pesos propios C7, CPU, un hilo de Torch, inferencia FP32. Para cada contexto
se midieron 12 cálculos por bloque, en orden completo/último/último/completo, con
calentamiento antes de cada bloque. La comparación usa el promedio de ambos
bloques de cada variante. Se conservaron los tiempos individuales y de CPU.

| Tokens de contexto | Antes, ms por cálculo | Ahora, ms por cálculo | Aceleración |
| ---: | ---: | ---: | ---: |
| 75 | 55,97 | 28,14 | 1,99× |
| 203 | 107,54 | 51,02 | 2,11× |

Son medidas locales del cálculo del siguiente token, no un porcentaje universal
de aceleración ni un ensayo de carga del producto.

## Conservación del comportamiento

Las 40 respuestas de desarrollo de C7 se reprodujeron exactamente, conservando
el historial realmente generado y las condiciones anteriores: temperatura 0,
semilla 42 y hasta 32 tokens nuevos. Los scores de ambas proyecciones coinciden
dentro de tolerancia FP32; la diferencia absoluta máxima observada fue menor
que 0,000009. No se afirma igualdad bit a bit de todos los scores posibles.

La prueba nueva compara generación completa y proyección final con y sin cabeza
de copia, tanto codiciosa como muestreada con semilla fija. Pasaron además las
pruebas pertinentes del modelo, runtime y cabeza de copia. El soporte de
respuestas largas publicado en `a1a283c` se conserva.

La búsqueda con ocho continuaciones también se ensayó separadamente para
investigar errores tempranos. No mejoró el número de respuestas completas en
el pequeño diagnóstico semántico. Esa variante no forma parte de esta corrección.

La evidencia y el script de medición están en
`docs/final-position-projection-evidence-20261012/`.
