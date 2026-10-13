# Estados completos de entrenamiento

Los paquetes conservan checkpoints nativos completos de la integración: pesos, Adam, programador, RNG, tokenizador y estado. El seleccionado y el último checkpoint se mantienen separados; el paso más reciente no es necesariamente la selección de validación.

| Paquete | Selección | Último |
| --- | ---: | ---: |
| [history-generated](history-generated/README.md) | 384 | 512 |
| [natural-v7](natural-v7/README.md) | 128 | 512 |
| [natural-c7](natural-c7/README.md) | 128 | 512 |

Las guías restauran a un directorio nuevo sin entrenar; la continuidad exige el corpus exacto y la configuración guardada. Los caches mmap se reconstruyen desde los corpus publicados. Cada archivo pesa menos de 100 MiB.

[Modelos de inferencia](../../models/conversation-integration-20261013/README.md). [Recibos de las etapas](../../docs/conversation-integration-20261013/training/README.md). Ningún paquete acredita conversación aceptada.
