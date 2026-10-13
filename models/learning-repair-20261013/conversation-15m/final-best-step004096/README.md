# BEST final de la etapa SFT

La etapa terminó en el paso 8.192 y conservó BEST en el paso 4.096, con NLL de
validación 1,7110848086831454. El artefacto final es idéntico, byte por byte, al
[artefacto intermedio completo](../intermediate-step004096/). Carga ese directorio
con el cargador normal de artefactos.

[artifact-reference.json](artifact-reference.json) identifica sus cuatro
archivos, hashes, modelo y checkpoint. Este directorio conserva una referencia
sin duplicar los pesos. La [entrega final](../../../../docs/learning-repair-20261013/conversation-15m-final-delivery.md)
distingue la exposición de BEST de la de LAST y documenta su recuperación.

El candidato conserva `productionAccepted=false`; NLL y checkpoints completos
no acreditan conversación aceptada.
