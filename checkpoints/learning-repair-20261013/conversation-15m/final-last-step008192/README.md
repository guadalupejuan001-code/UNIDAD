# LAST nativo del cierre SFT

El paquete [step-000008192-82822b3e01a7488dbb1141a1174ebb94](step-000008192-82822b3e01a7488dbb1141a1174ebb94/)
conserva los cinco archivos nativos y recompone 182.409.967 bytes mediante nueve
partes de hasta 32 MiB. Sus manifiestos y todos los hashes fueron verificados.
El estado mantiene los 8.192 pasos, Adam, RNG, orden de ventanas, configuración,
scheduler y los punteros originales BEST/LAST.

BEST permanece en el paso 4.096. Su
[paquete existente](../intermediate-step004096/step-000004096-37b0b60acf4546d398078b3c407e600d/)
también fue verificado y debe conservarse para resolver ambos punteros.

La [entrega final](../../../../docs/learning-repair-20261013/conversation-15m-final-delivery.md)
incluye instrucciones de recomposición y condiciones de recuperación. Esta
entrega verificó el transporte de bytes; no cargó el optimizador ni ejecutó una
restauración del entrenador.
