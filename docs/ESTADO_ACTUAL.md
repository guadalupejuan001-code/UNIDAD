# Estado actual de Unidad

Comprobado el 12 de octubre de 2026. **La conversación básica y la calidad competitiva siguen pendientes. C7 continúa como referencia experimental; no se promovieron candidatos.** Este estado actualiza la presentación del proyecto sin reemplazar pesos, respuestas ni recibos históricos.

## Modelo conservado y aceptación

C7 ya estaba publicado en `main`: [pesos](../models/conversation-reference-c7/weights.safetensors), [configuración](../models/conversation-reference-c7/config.json), [tokenizador](../models/conversation-reference-c7/tokenizer.json) y [manifiesto](../models/conversation-reference-c7/manifest.json). Tiene 6.983.937 parámetros con copia v1; su backbone tiene 6.967.296. BPE de 8.192 y contexto de 512 tokens. La [auditoría de once modelos](unidad-github-model-inventory-20261012.json) verificó los 44 archivos contra GitHub y sus manifiestos, sin inconsistencias. No acredita competencia conversacional.

La aceptación exige **36/40 completas, 8/10 diálogos completos y cero fallos críticos o técnicos**, mediante una prueba nueva reservada después de seleccionar el candidato. La meta parcial es llegar a un máximo de 14 fallidas de los 40 turnos conocidos sin superar seis críticas; tampoco se consiguió.

| Modelo / revisión primaria de desarrollo | Completas | Parciales | Fallidas | Críticas | Diálogos completos |
| --- | ---: | ---: | ---: | ---: | ---: |
| C7 | 7 | 5 | 28 | 6 | 0/10 |
| Control original128 | 8 | 1 | 31 | 7 | 0/10 |
| Relacional128 | 6 | 2 | 32 | 9 | 0/10 |

El cruce da control 9/8/23 con siete críticas y relacional 7/6/27 con ocho; ambos 0/10. Todas las etiquetas y discrepancias se conservan. En el factorial de 64 turnos reales, control 4/11/49 con ocho críticas y relacional 5/7/52 con trece; en 80 turnos de formulaciones/composiciones reservadas ambos alcanzan ocho completas, 17 críticas y 0/20 diálogos. Las rúbricas de esos subconjuntos tienen alcances distintos; no se suman ni comparan sus críticas como una sola evaluación. Las revisiones son de IA del proyecto, no humanas independientes. El [informe completo](https://github.com/guadalupejuan001-code/UNIDAD/blob/codex/conversation-wording-repair-20261012/docs/work-conversation-repair-20261012.md) conserva todas las respuestas.

## Código y propuestas

El evaluador de `main` se actualiza con la corrección ya probada: 40 turnos, requisito de diálogos completos, hashes de casos/rúbrica, historia real y rechazo de vacías o críticas incorrectas. También verifica identidad del manifiesto y consistencia del recorte. Pasaron 26 pruebas pertinentes. Sus condiciones históricas de generación 24/0/109 se conservan; las nuevas pruebas de diagnóstico utilizaron 160/0/42. Superar pruebas de software no acredita conversación.

Los PR siguientes permanecen en borrador, sin fusionarse como conjunto:

| PR | Alcance y estado comprobado |
| --- | --- |
| [1](https://github.com/guadalupejuan001-code/UNIDAD/pull/1) | Evaluador y diagnóstico inicial. Sus experimentos usan `b5fc677`; sus resultados son históricos y el parche de copia v2 sigue experimental, sin aplicar. La corrección del evaluador se incorpora aparte. |
| [2](https://github.com/guadalupejuan001-code/UNIDAD/pull/2) | Proyección en última posición y prueba de redacción. El candidato 448 obtuvo 13/12/15, seis críticas y 2/10 en desarrollo; en 40 turnos nuevos 3/5/32, quince críticas y 0/10. Rechazado; cifras publicadas por ese PR, sin restaurar sus pesos aquí. |
| [3](https://github.com/guadalupejuan001-code/UNIDAD/pull/3) | Contraste relacional desde C7: dos de 128, retención rechazada, referencia conservada. Código experimental `8b466cc`; no se promueve por menor pérdida. |
| [4](https://github.com/guadalupejuan001-code/UNIDAD/pull/4) | Exportador Android del candidato 448 del PR2. Paridad numérica/textual comprobada allí; `androidDeviceTested=false`. No acredita conversación ni prueba en dispositivo físico. |

Los cambios de motor y exportador de PR2/4 siguen como propuestas; no se mezclan con las versiones que produjeron las métricas anteriores. C7 no pasa a ser automáticamente el modelo de todos los productos: los anfitriones seleccionan su artefacto explícitamente.

## Pesos y checkpoints descargables

El [paquete en GitHub](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012) añade los candidatos original128/relacional128 y sus estados completos a la referencia C7. Está dividido en seis partes consecutivas de hasta 64 MiB. El script verifica cada parte y reconstruye exactamente `Unidad-Contraste-Relacional-Experimental.zip`:

```bash
git clone --single-branch --branch artifacts/unidad-contraste-relacional-20261012 https://github.com/guadalupejuan001-code/UNIDAD.git unidad-con-pesos
cd unidad-con-pesos
python weights-packages/unidad-contraste-relacional-20261012/reconstruir.py
```

SHA-256: `5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`; 339.107.857 bytes; 370 archivos, 290 objetos únicos y ocho checkpoints completos. Incluye tres identidades propias de modelo y cinco alias portables. Se verificaron los objetos/tamaños remotos, la reconstrucción exacta, la restauración en un directorio nuevo y la consola C7 con cuatro turnos reales. No hay assets de release publicados; la vía usada fue Git.

El paquete **no contiene** `Unidad-Pesos-Diagnostico.zip` (SHA 4ef475…), `Unidad-Prueba-Propia.zip` del PR2 (SHA 4a68aa…) ni el binario `.unidad` del PR4 (SHA 2de098…). Esos archivos no están disponibles en este workspace y no se presentan como restaurados. El PR4 publica el exportador y su evidencia, no su binario.

Para probar C7 desde un clon con las dependencias de conversación instaladas:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

## Procesos y presupuestos

En este entorno comprobado no hay entrenamiento ni generación activos. Los presupuestos originales de 24 horas y las dos horas adicionales están cerrados. La comparación nueva acabó sus 128 pasos por brazo; no se amplía automáticamente. Conservar pesos, optimizador o RNG no significa que el modelo siga aprendiendo. No se afirma haber inspeccionado procesos de otros entornos.

Las descripciones de etapas que usaban tiempo restante, pruebas antiguas de 30 turnos y recibos fechados se mantienen como historia; no renuevan presupuestos ni sustituyen la aceptación vigente. El [estado estructurado](unidad-current-status.json) identifica fuentes, pesos, métricas y disponibilidad exactas.
