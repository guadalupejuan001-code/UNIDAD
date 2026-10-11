# Base española experimental de Unidad v1

Esta instantánea conserva el preentrenamiento completado de un Transformer propio
iniciado desde pesos aleatorios, sin pesos preentrenados externos. Tiene
**6.967.296 parámetros**, BPE de **8.192 tokens**, contexto de **512 tokens**,
seis capas, dimensión 256 y ocho cabezas de atención. Los archivos nativos
`config.json`, `tokenizer.json`, `manifest.json` y `weights.safetensors` se
conservaron exactamente, sin modificar sus bytes ni el manifiesto.

El preentrenamiento completó **6.000 pasos** con objetivo `causal_lm`: predicción
de tokens de texto y EOS. La instantánea seleccionada corresponde al paso 6.000;
la mejor pérdida de validación sobre TEXT fue **4.7754543702337795**. Es una
medición de validación de lenguaje, no de TEST ni de conversación competente.
No se evaluó TEST para conservar estos archivos.

Esta es una **base experimental de español**. Todavía no se ha aceptado como
conversación competente o trabajador autónomo. El ajuste conversacional posterior
completó otros 6.000 pasos, con los mismos pesos iniciales propios y tokenizador,
y un optimizador nuevo. Su instantánea está separada en
[conversation-language-v7](../conversation-language-v7/README.md); los cuatro
archivos nativos de esta base permanecen intactos.

En 40 turnos nuevos del sucesor v7, la revisión primaria y la cruzada coincidieron
en **2 respuestas completas, 3 parciales y 35 fallidas**, con **0 de 10 diálogos
completos**, sin errores de ejecución ni truncamiento. Son revisiones de
asistentes de IA del proyecto, no de un panel externo. V7 no superó el criterio
de 36 de 40 respuestas completas, 8 de 10 diálogos completos y cero fallos
críticos; no se aceptó ni promovió como modelo predeterminado y no se consultó
TEST. El [informe de resultados](../../docs/unidad-conversation-language-v7-results.json)
conserva los recuentos originales de 13 y 12 fallos críticos y la unión de 14
turnos críticos confirmada por ambos asistentes tras la reconciliación, sin
cambiar las 40 calificaciones ni el rechazo. Estos resultados pertenecen al
sucesor SFT, no a una evaluación
conversacional de esta base de texto. Todos los modelos históricos se conservan.

La [guía de preentrenamiento](../../docs/language-pretraining.md) explica los
objetivos, el formato y la separación entre etapas. El
[registro público de fuentes y preparación](../../docs/unidad-spanish-language-base-v1.json)
conserva revisiones, hashes, particiones y atribución. No se incorporó el historial
personal del creador al corpus de esta base.

## Fuentes y declaraciones de licencia

Los textos españoles proceden de
[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
revisión `fdf72ae0827c1cda404aff25b6603abec9e3399b`, cuya declaración de licencia
es [Apache-2.0](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE),
y de Wikipedia en español, configuración `20231101.es`, revisión fija
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.

La [ficha fija de Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/raw/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)
declara CC BY-SA 3.0 y GFDL. La
[documentación legal actual de Wikimedia](https://dumps.wikimedia.org/legal.html)
declara CC BY-SA 4.0 y GFDL. Se conservan ambas declaraciones por separado:
no certifican derechos artículo por artículo ni que todos los textos tengan
licencia dual. La preparación mantiene metadatos de atribución como título,
URL de artículo, URL de historial, identificador de página, fecha de snapshot,
revisión y SHA-256 del texto, según el registro enlazado.

Estas son declaraciones sobre las fuentes y su atribución; este README no
certifica derechos ni una licencia de distribución de los pesos entrenados.

Modelo: `unidad-conversation-9077c0ede2a5564e671f44e10762933287c5c9ba5efc2c070e5a23e071d025a0`.

SHA-256 de los pesos: `1b4e52bae66e2ff6a3d56730b8da4fd4f69d2376d1fe5ce7b79d1408aa82d3db`.
