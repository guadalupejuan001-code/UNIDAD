# Candidato conversacional experimental de Unidad v7

Esta instantánea conserva el ajuste conversacional SFT completado de un
Transformer propio de **6.967.296 parámetros**: BPE de **8.192 tokens**, contexto
de **512 tokens**, seis capas, dimensión 256 y ocho cabezas de atención.
`config.json`, `tokenizer.json`, `manifest.json` y `weights.safetensors` se copiaron
exactamente desde la instantánea seleccionada, sin modificar sus bytes.

La procedencia es **pesos aleatorios propios → 6.000 pasos de preentrenamiento
de español → 6.000 pasos de SFT**. El SFT comenzó desde los pesos de
[spanish-foundation-v1](../spanish-foundation-v1/README.md), conservó su
configuración y tokenizador y usó un optimizador y un programador nuevos.
Es una etapa posterior desde un checkpoint propio; no es una reanudación del
optimizador del preentrenamiento ni usa pesos preentrenados externos.

El objetivo SFT fue `assistant_last_reply`, con supervisión de la última
respuesta del asistente. Se seleccionó el paso **6.000** por validación; la NLL
de validación registrada es **1.6373902535790683** (aproximadamente **1,63739**).
**TEST no se ejecutó** para entrenar, seleccionar o conservar esta instantánea.
La pérdida de validación no demuestra conversación competente.

Este es un **candidato experimental**, sin aceptación conversacional ni de
trabajador autónomo. En 40 turnos de desarrollo nuevos, repartidos en diez
diálogos, la revisión primaria y la cruzada coincidieron: **2 respuestas
completas, 3 parciales y 35 fallidas; 0 de 10 diálogos completos**. Se generaron
los 40 turnos sin errores de ejecución ni truncamiento. Las revisiones fueron
realizadas por asistentes de IA del mismo proyecto, no por un panel externo.

No superó el criterio fijado: al menos **36 de 40 respuestas completas**, **8 de
10 diálogos completos** y **ningún fallo crítico**. La primaria contó 13 fallos
críticos y la cruzada 12. Tras comparar las revisiones, ambos asistentes
confirmaron una unión de **14 turnos con fallos críticos**, con 11 compartidos.
Se conservan los recuentos originales y las 40 calificaciones; la reconciliación
no cambia el rechazo de la aceptación. No se consultó TEST ni se promovió v7
como modelo predeterminado. Los 15 aciertos de v6 en 22 turnos conocidos
corresponden a otra evaluación y no constituyen una comparación equivalente.

El [informe público de resultados](../../docs/unidad-conversation-language-v7-results.json)
conserva los hashes, la configuración y los límites. SHA-256 de las salidas
reales revisadas:
`58291a3f0514fa19c76f92a59b92c55053e5ad4bfcfee1a09a3544ab0124ba9e`.
Copiar los pesos y verificar su integridad no valida capacidades. Las
comparaciones privadas posteriores de aprendizaje son experimentos separados;
no se han adoptado ni incorporado sus pesos en este directorio. La base española
y todos los modelos históricos se conservan.

La [guía de preentrenamiento](../../docs/language-pretraining.md) describe las
etapas y sus contratos. El
[registro público de fuentes y preparación](../../docs/unidad-spanish-language-base-v1.json)
conserva revisiones, hashes, particiones y atribución. El SFT combina conversación
pública de OpenAssistant con diálogos ficticios redactados para el proyecto,
incluido el [currículo de diversidad](../../docs/unidad-natural-diversity-v2.md).
El historial personal del creador no se incorporó como corpus.

## Fuentes y declaraciones de licencia

La base de español hereda textos de
[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
revisión `fdf72ae0827c1cda404aff25b6603abec9e3399b`, cuya declaración es
[Apache-2.0](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE),
y de Wikipedia en español, configuración `20231101.es`, revisión
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.

La [ficha fija de Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/raw/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)
declara CC BY-SA 3.0 y GFDL; la
[documentación legal actual de Wikimedia](https://dumps.wikimedia.org/legal.html)
declara CC BY-SA 4.0 y GFDL. Son declaraciones distintas, conservadas con la
atribución enlazada; no certifican derechos artículo por artículo, licencia
dual de todos los textos ni derechos de distribución de los pesos entrenados.

Modelo: `unidad-conversation-9081da24a3a73e748887dd66ee937798ef6808579ed5cdb78c3b27b70639cd9d`.

SHA-256 de los pesos: `dc4ba778deb188fabdf66b59aa2457578c720040bf0e36666ed9e234f1ed288e`.

SHA-256 del manifiesto nativo: `0dba7d9ddaedb4fba70f490a39ccfed85da12cec6f188b1d891ab6ac4e1aec11`.
