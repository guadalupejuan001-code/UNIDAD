# Datos reproducibles del aprendizaje de UNIDAD

Los dos archivos de entrenamiento contienen únicamente TRAIN y VALIDATION. El TEST reservado no se publica en este paquete ni se usa en el entrenamiento. No se añadieron conversaciones personales del creador, etiquetas de evaluación, modelos externos o servicios de pago.

| Archivo | TRAIN | VALIDATION | SHA256 |
|---|---:|---:|---|
| `causal-base/causal-spanish.train-validation.jsonl` | 10.735 | 1.351 | `5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17` |
| `sft/sft.train-validation.jsonl` | 18.588 | 1.704 | `4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510` |

La base causal contiene texto original de Wikipedia y mensajes públicos revisados de OpenAssistant. Wikipedia es la instantánea española **20231101.es**, con revisión del conjunto [`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa). No representa noticias de 2026 ni garantiza actualidad factual. El manifiesto original fija las 52 respuestas de páginas de filas, órdenes 541–592, sus offsets y hashes. Los grupos se asignan por identificador del artículo, mediante la misma función determinista de partición del repositorio; ningún artículo se mueve de TEST a TRAIN.

OpenAssistant usa revisiones fijas:

| Fuente pública | Revisión | SHA256 de los árboles JSONL descargados |
|---|---|---|
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b) | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7` |
| [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2/tree/179dd21fc55192153d94adb0e0ce8f69e222bf75) | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | `7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b` |

Sus importaciones canónicas tienen SHA256 `96dc38a0aa4434516840c67b75c5ddabbfb4c0daf1e4b46cca274bc1c1e36825` y `161b8c4bda185889d1e77b1581eccd6b24810f37292e30db05edc9a1ef07b5b9`. Los identificadores, particiones familiares, mensajes y revisiones se preservan; los IDs repetidos de OASST1/OASST2 se cuentan una sola vez. El importador excluye mensajes borrados, sin revisión positiva, marcados como privados/spam y sintéticos externos. Las reglas explícitas de preparación excluyen familias con identidad externa del asistente y, para SFT, afirmaciones de acciones externas sin resultados de herramientas. Esas reglas conservadoras no son una revisión factual exhaustiva.

La unión SFT combina 8.657 prefijos humanos públicos, 10.880 prefijos de composiciones ficticias y 760 prefijos manuales redactados por asistentes de IA. Cinco conversaciones idénticas del mismo fold se retiran. Cero familias cambian de fold; cero respuestas se reescriben durante la unión. El humano representa 41,30 % de TRAIN. La longitud se midió antes de la unión con el BPE2048 propio, SHA256 `848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720`; 512 tokens de entrada como máximo. No se cargaron pesos durante la unión.

La supervisión `assistant_last_reply` usa como objetivo positivo solamente la última respuesta del asistente y EOS. La revisión editorial de los autores excluyó siete prefijos erróneos de TRAIN y uno de VALIDATION; los prefijos correctivos posteriores conservan literalmente la historia original. La comprobación final por IDs se conserva en `sft/final-target-exclusion-verification.json`. Esta revisión fue realizada por asistentes de IA del proyecto, antes del entrenamiento SFT, sin casos reservados. No es una revisión humana independiente ni demuestra competencia del modelo.

Para reconstruir la unión se usa la implementación pública canónica `examples/merge_learning_sft.py`, SHA256 `bce38d29a2c1ff027d14275b0fb9daf71c61a492ae815540b3f799e82d68d360`. Conserva literalmente los bytes del helper de preparación original, archivado como `private/learning-repair-20261013/sft-union-source/merge_learning_repair_sft.frozen.py`; ese archivo privado no es necesario para ejecutar la copia pública. Requiere como entradas congeladas:

- Humano SFT: SHA256 `24770f0b2bb1381fd70ab3800235f6d03e72f02128addfc424bcbf8b83d166fd`, con `source.kind=public-human-reviewed`.
- Composiciones: SHA256 `be4d7d84faaec0afe2cc797f3f0a5fef9b60dbe3b420f4b535daa805f0943394`, con `source.kind=authored-synthetic`.
- Manuales ya filtrados: SHA256 `6b1e805992ecca278ebb80b242ebfd46407d567b3a41f952e41953273044d93e`, con `source.kind=authored-synthetic`.
- Recibo de exclusiones: SHA256 `17137c69a8f98345cc90a30209a0f5225d064c5730594aed95238664f9b99279`.

El helper acepta `--input PATH SHA256 KIND` tres veces, `--target-exclusion-receipt PATH`, `--tokenizer-sha256 SHA` y `--output DIRECTORIO_NUEVO`. Los manifiestos originales conservan las rutas de preparación históricas; las rutas portables actuales son las de este paquete. Las comprobaciones de hashes se aplican al contenido literal, no a una reescritura de sus recibos.

Las licencias y límites de calidad se detallan en `LICENSE-DATA.md`. Los filtros no certifican cada hecho, las plantillas sintéticas repiten estructuras, y más datos no garantizan mejor conversación. La generación de desarrollo y la evaluación reservada siguen siendo actividades distintas de la preparación de estos datos.
