# Ampliación de conversación E2/E3: resultado comprobado

La ampliación autorizada terminó sin demostrar conversación básica ni competitiva en español. C7 conserva el mayor número de respuestas completas entre los candidatos propios de esta comparación: 7 de 40. El candidato E3 publicado obtuvo 3 de 40, ningún diálogo completamente correcto y 12 fallos críticos en cada revisión. Las revisiones son de agentes de IA del proyecto.

| Candidato | Primaria C/P/F | Cruzada C/P/F | Críticos primaria/cruzada | Diálogos completos |
| --- | --- | --- | --- | --- |
| C7 | 7/5/28 | 7/7/26 | 6/8 | 0/10 |
| e2-step2048-live-v1 | 3/8/29 | 3/8/29 | 7/9 | 0/10 |
| e2-step4096-live-v1 | 5/6/29 | 6/10/24 | 6/4 | 0/10 |
| e2-step6144-stopped-v1 | 2/5/33 | 3/7/30 | 7/8 | 0/10 |
| e2-step6208-stopped-v1 | 3/5/32 | 3/7/30 | 9/9 | 0/10 |
| e3-step1024-live-v1 | 0/5/35 | 0/7/33 | 10/11 | 0/10 |
| e3-step2048-live-v1 | 2/8/30 | 2/11/27 | 7/8 | 0/10 |
| e3-step3072-v1 | 3/5/32 | 3/3/34 | 12/12 | 0/10 |

El criterio básico sigue siendo 36 respuestas completas de 40, 8 diálogos de 10 y cero fallos críticos. Es necesario, pero no suficiente para la misión competitiva. Los 40 turnos son desarrollo conocido; no se abrió la evaluación final reservada ni TEST.

E2 se detuvo cooperativamente en 6.208 de los 8.192 pasos previstos al cambiar de hipótesis; su supervisor terminó con código 1 por la comprobación original de pasos. E3 terminó el punto final de 3.072 pasos fijado antes de entrenar, con código 0. Ambos hijos fueron recogidos, sin parada forzada. Los checkpoints nativos y optimizadores están preservados; no queda entrenamiento activo.

El [informe JSON](unidad-conversation-supplement-e2-e3-results.json) conserva SHA-256, respuestas reales, historial, revisiones, estado y límites. El [README de E3](../models/conversation-initial-availability-e3/README.md) incluye el comando comprobado; las cuatro respuestas reales de esa consola coinciden exactamente con el candidato evaluado.

La referencia externa Qwen2.5-0.5B-Instruct es pública, Apache 2.0, revisión fijada. Se ejecutó sólo para comparar sobre los mismos 40 turnos conocidos y el mismo sistema, con historia propia real, generación greedy y 32 tokens por su tokenizador. Difieren tokenizador, adaptador, capacidad, contexto configurado y número de hilos; no se afirma igualdad de cómputo ni velocidad. Sus respuestas nunca entraron en el entrenamiento ni sustituyen el runtime o los pesos de Unidad. Esta muestra limitada no demuestra competitividad.

Qwen, primaria C8/P7/F25; cruzada C10/P4/F26; críticos 15/20; 0/10 diálogos completos.

Una [prueba adicional de desarrollo cotidiano](unidad-conversation-supplement-e2-e3-evidence/everyday-probe/actual-generation.json) pidió explicaciones, un ejemplo, una reformulación, una invitación y correcciones. C7 no cumplió completamente ninguna de sus seis peticiones, aunque se permitió hasta 96 tokens. Las respuestas y la revisión de IA del proyecto se conservan sin cambios; esta prueba no se mezcló con el entrenamiento ni con la evaluación final reservada.
