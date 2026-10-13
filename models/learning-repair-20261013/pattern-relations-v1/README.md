# Pesos propios del control de patrones de UNIDAD

Esta carpeta contiene dos versiones de pesos del mismo Transformer propio de
UNIDAD, con 15.769.344 parámetros, BPE de 2.048 tokens y contexto de 2.048.
No contiene un modelo preentrenado de otra organización.

- `reference-wiki1`: padre propio cerrado, `c0ee5f24…`, posterior al primer lote
  automático de Wikipedia y su ajuste conversacional. Son sus cuatro archivos
  originales, copiados y verificados byte a byte.
- `candidate-best`: `af515755…`, checkpoint del paso 128 seleccionado por
  validación después de ejecutar 512 actualizaciones del control. Es experimental.
- `native-best128-parts`: cinco archivos originales del checkpoint BEST, incluido
  el optimizador, repartidos en partes de como máximo 32 MiB. Se restauraron y
  comprobaron sus hashes; no se entregan aquí los archivos completos de LAST 512.

El nombre físico `v1` se conserva en las rutas fijadas antes de ejecutar. El
paquete de datos y criterio es `pattern-relations-v2`; conserva los mismos
ejemplos iniciales con el preregistro alineado antes del entrenamiento.
Los identificadores no son productos o IAs externas: el proyecto es UNIDAD.

El candidato aprendió once de doce respuestas TRAIN, pero sólo una de doce
respuestas nuevas fue completa según la revisión semántica de IA y fallaron
los dos diálogos. No superó la aceptación ni sustituye automáticamente a la
referencia. [Resultado, hashes e instrucciones](../../../docs/learning-repair-20261013/pattern-relations-v2/README.md).
