# Modelos experimentales de la integración del 13 de octubre

Estos seis artefactos conservan las selecciones por validación de los entrenamientos propios. Todos siguen experimentales; la aceptación conversacional permanece pendiente. C7 conserva su directorio histórico.

| Modelo | Pasos realizados | Paso conservado |
| --- | ---: | ---: |
| [Español humano desde V7](v7/README.md) | 256 | 256 |
| [Español humano desde C7](c7/README.md) | 256 | 256 |
| [Historia ideal desde C7](history-ideal/README.md) | 512 | 512 |
| [Historia real desde C7](history-generated/README.md) | 512 | 384 |
| [Diálogos diversos desde V7](natural-v7/README.md) | 512 | 128 |
| [Diálogos diversos desde C7](natural-c7/README.md) | 512 | 128 |

Cada directorio contiene configuración, tokenizador, manifiesto y pesos nativos con hashes comprobados. La pérdida de validación elige una instantánea dentro de cada etapa; no demuestra calidad conversacional.

[Recibos y diagnóstico](../../docs/conversation-integration-20261013/training/README.md). [Estados completos para continuar](../../checkpoints/conversation-integration-20261013/README.md).
