# Cierre de LANG1 sobre español LATAM v3

LANG1 terminó en **2.048 actualizaciones del optimizador**, con tiempo acumulado
de **1.931,1925520530021 segundos**. Se comprobó que el trabajador original había
salido y que el recibo final del runner estaba completo. El estado conserva
`status=stopped`, motivo nativo **`signal`** y motivo cooperativo
**`absolute_step_boundary`**.

El umbral configurado era 1.800 segundos. El tiempo registrado lo sobrepasó en
**131,19 segundos**: la parada cooperativa termina el trabajo pendiente en el
límite, incluida validación y guardado. Los recibos no desglosan cuánto consumió
cada operación. No se presenta el umbral como un límite estricto de pared ni se
cambia el motivo nativo a `total_time_budget`.

BEST y LAST son checkpoints distintos del paso 2.048. Comparten exactamente
los pesos, SHA256
`b8856eed20322b4c0011b274fe582a3b9ff6b1ab57d874cb300420a617b5dc69`,
el optimizador/RNG y el tokenizador; sus estados y manifiestos son diferentes.
El modelo propio seleccionado es `unidad-conversation-a8c25b012d4cbf1da6b818983dc424ec219b93b7793115bb5d6474d8ef3087f8`,
con contexto 2.048 y NLL de validación **3,2358626570786604**. Procede de ACTION256
propio, creó un optimizador y scheduler nuevos y entrenó los 15.769.344 parámetros.

El [recibo principal](wide-latam-v3-language-1-final-delivery.json) conserva
configuración, procedencia, hashes, código y textos originales de los metadatos
congelados. Sus campos `utf8Text` reconstruyen exactamente los bytes originales
de estado, diseño, recibo del runner y manifiesto de caché. La exportación del
artefacto de inferencia completo corresponde a otra entrega del agente principal.

## Exposición TRAIN real

El [registro exacto](wide-latam-v3-language-1-final-target-exposure.json) deriva
los totales del índice TRAIN, sus etiquetas, el orden nativo y el RNG completo.
Su objetivo es `causal_lm`, reducción `token_mean`, texto y EOS supervisados,
con ventanas contiguas sin solapamiento. No leyó el corpus ni las cachés
VALIDATION/TEST. La revisión independiente de IA confirmó los mismos totales y
la reproducción exacta del barajado con semilla 20261013.

| Posiciones ejecutadas por BEST y LAST | Total |
| --- | ---: |
| Ventanas consumidas una vez | 4.096 |
| Tokens de entrada sin padding | 1.445.480 |
| Posiciones supervisadas, incluido EOS | 1.445.480 |
| Posiciones EOS supervisadas | 3.921 |
| Posiciones supervisadas sin EOS | 1.441.559 |

El lote lógico es dos y el microbatch máximo uno. La corrida terminó durante la
primera época: el cursor nativo es 4.096 de **10.206 ventanas TRAIN**. Quedan
**6.110 ventanas sin consumir** en esta etapa. El inventario completo contiene
3.594.399 posiciones supervisadas y 9.746 EOS, que no se cuentan como ejecutados.
El ledger conserva longitudes, etiquetas, EOS y multiplicidad de consumo para
cada ventana. No permite deducir cuántos registros, palabras o fuentes completos
se cubrieron ni cuánto conocimiento adquirió el modelo.

## Reponer BEST y LAST

[LAST](../../checkpoints/learning-repair-20261013/wide-latam-v3/language-1-final/LAST/step-000002048-4ce480718b1142e793cc65c3fc361e66/)
recompone **189.437.568 bytes nativos** mediante nueve partes de hasta 32 MiB.
Se verificaron los hashes de cada parte y de los archivos completos recorriendo
las partes en orden. Conserva pesos, Adam/RNG, scheduler, configuración, orden de
ventanas y los nombres nativos.

[BEST-reference](../../checkpoints/learning-repair-20261013/wide-latam-v3/language-1-final/BEST-reference/step-000002048-4fca708c21bf45889075375241a00b00/)
conserva sus propios `state.json` y `manifest.json` y referencia las partes
idénticas de LAST. Los cinco flujos de reconstrucción se comprobaron contra los
hashes del BEST original, evitando duplicar pesos y Adam/RNG. Este directorio de
referencia no se carga directamente como un checkpoint.

Desde la raíz del repositorio, con destinos nuevos:

```bash
mkdir -p private/wide-latam-v3-language-1-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/wide-latam-v3/language-1-final/LAST/step-000002048-4ce480718b1142e793cc65c3fc361e66 \
  private/wide-latam-v3-language-1-recovery/checkpoints/step-000002048-4ce480718b1142e793cc65c3fc361e66
python3 docs/learning-repair-20261013/restore_action_native_reference.py \
  checkpoints/learning-repair-20261013/wide-latam-v3/language-1-final/BEST-reference/step-000002048-4fca708c21bf45889075375241a00b00 \
  private/wide-latam-v3-language-1-recovery/checkpoints/step-000002048-4fca708c21bf45889075375241a00b00
```

El helper de referencias existente también recompone estos cinco archivos
nativos; su nombre conserva su primera aplicación ACTION. Son instrucciones de
recomposición, aún no ejecutadas. No cargan modelos ni arrancan entrenamiento.
La recuperación del entrenador también requiere el corpus con SHA256
`81c414480d0baf36b7784f808ba8d4759a6b7ef706eeaabdcb74d6e919cb8a81`,
el diseño y código congelados. Conserva los 2.048 pasos y el tiempo ya consumido;
no reinicia sus límites.

La entrega usó biblioteca estándar, una CPU, prioridad `nice +10`, espacio de
direcciones limitado a 512 MiB y operaciones inferiores a 120 segundos. No
importó Torch, interpretó pickle, ejecutó inferencia, entrenamiento, pruebas o
restauración, leyó evaluaciones reservadas ni modificó código, trabajador,
supervisor o LANG2. La revisión es de IA del proyecto.

La preparación local no verifica una subida a GitHub. Pérdida, exposición y
archivos íntegros no acreditan conversación aceptada: se conserva
`conversationAccepted=false`.
