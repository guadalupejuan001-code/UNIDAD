# Integración del motor y exportación

La reconciliación incorpora el evaluador de PR1/3, ya presente en `main`, la
proyección de vocabulario de PR2 sólo para la posición solicitada al generar y
el exportador de PR4. Conserva el núcleo `ConversationEngine` y los pesos C7.
Los documentos históricos importados mantienen sus bytes y fecha; sus métricas
no se atribuyen al código nuevo. El parche `source-pointer-v2.patch` permanece
como evidencia experimental y no se aplica.

[branch-reconciliation.json](../branch-reconciliation.json) identifica las ocho
ramas, sus cabezas y los 324 documentos importados. La rama de paquetes conserva
los experimentos descargables en su ubicación original. No se mezclan los ZIP
históricos con el código ni se promueven candidatos rechazados.

## Comprobación local

[final-projection.json](final-projection.json) conserva cuatro respuestas C7
idénticas a las archivadas, condiciones, tokens y hashes. En CPU con un hilo, la
medición de cálculo del siguiente token observó factores 2,03 y 1,91 para
contextos de 75 y 203 tokens; no mide aceleración de una conversación completa.
La diferencia máxima de logits fue 3,81e-6. La optimización no acredita mejora
semántica.

[c7-mobile-parity.json](c7-mobile-parity.json) registra siete casos de logits y
cuatro generaciones greedy iguales entre PyTorch y ONNX Runtime, con error
máximo 2,24e-5. Se ejecutó en CPU Linux; no se probó en un teléfono Android.
El exportador rechaza sobrescrituras y cambios del modelo antes de publicar.

```sh
python -m pip install -e '.[conversation,mobile]'
python scripts/export_unidad_mobile.py \
  --model models/conversation-reference-c7 \
  --output /tmp/c7.unidad --evidence /tmp/c7-parity.json
```

Exportar un modelo no lo entrena ni demuestra comprensión. El recibo identifica
el binario producido en la comprobación; los pesos nativos C7 ya están en Git.
GitHub Actions ejecuta las pruebas con los extras de conversación y móvil.
