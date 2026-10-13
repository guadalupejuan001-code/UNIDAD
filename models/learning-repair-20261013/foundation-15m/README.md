# Base propia de español: exportaciones experimentales

`final/` conserva la exportación de inferencia al cerrar la etapa causal propia
por límite de tiempo. Completó **8.914 updates**, tres épocas completas y
17.312 ventanas de la siguiente: **22.906.669 posiciones objetivo** procesadas
en **71.297 ventanas**. El contador nativo final registra **7.337,688 segundos**;
el umbral de parada era 7.200 segundos y el cierre añadió validación y guardado.
No completó el máximo configurado de 12.000 updates.

La base final identifica el modelo
`unidad-conversation-ee24651fb8cc3a01e1acbb57db5f40222a6b9b5fe150dbff7186b961945fa398`.
Su selección por validación causal quedó en step 8.914, con NLL
**2,981873986010036**. El artefacto conserva cuatro archivos exactos, con
60.784.468 bytes; los
[recibos de entrega](../../../docs/learning-repair-20261013/foundation-15m-final-delivery.json)
y [exposición exacta](../../../docs/learning-repair-20261013/foundation-15m-final-target-exposure.json)
conservan hashes, procedencia y contabilidad. El estado nativo estaba cerrado y
su worker muerto antes de copiar. No se ejecutó inferencia para esta entrega.

La finalización de la etapa causal y su pérdida no acreditan conversación
competente. La base permanece experimental, sin aceptación de producción y
sin sustituir C7. El ajuste conversacional y su evaluación se documentan por
separado.

## Primera exportación conservada

`intermediate-step004096/` conserva la primera exportación de inferencia de la
pausa de la etapa causal `foundation-15m`. Es un Transformer propio de
15.179.520 parámetros iniciado desde pesos aleatorios, con BPE de 2.048 tokens
ajustado sólo con TRAIN y contexto de 512. No usa pesos preentrenados ajenos.

La exportación contiene exactamente cuatro archivos: configuración,
tokenizador, pesos y manifiesto, con un total de **60.784.469 bytes**. Sus hashes
coincidieron antes de copiar, en la copia y después de copiar. El
[recibo de entrega](../../../docs/learning-repair-20261013/foundation-15m-intermediate-step004096-delivery.json)
conserva esos hashes y la procedencia.

Identifica el modelo
`unidad-conversation-1e46abcba0e6d5cbf25380bc98b92190ef556afbd5acd3463c92e399f32e3666`,
seleccionado en step 4.096 por NLL de validación **3,4665581824610308**.
Es una base lingüística causal intermedia; todavía no acredita conversación
competente y no reemplaza C7. Su exportación durante la pausa no representó
el cierre de la etapa. El manifiesto conserva `experimental=true` y
`productionAccepted=false`.

Las dos exportaciones sirven para cargar pesos en inferencia y conservan
versiones históricas. No contienen Adam ni RNG para reanudar. Los checkpoints nativos
recuperables se distribuyen por separado, según
[checkpoint-packaging.md](../../../docs/learning-repair-20261013/checkpoint-packaging.md).
La publicación remota y las respuestas conversacionales se comprueban por
separado; no se atribuyen a esta operación de copia.
