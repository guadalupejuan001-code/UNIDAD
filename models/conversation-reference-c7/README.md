# Unidad C7: referencia experimental

Estos son pesos propios de Unidad. **Todavía no cumplen la aceptación de conversación básica.** C7 se conserva como referencia experimental; los candidatos posteriores examinados no superaron sus criterios de promoción. Esa selección no significa que C7 sea el mejor modelo en todas las métricas o conversaciones.

La revisión primaria del proyecto obtuvo 7 respuestas completas, 5 parciales y 28 fallidas de 40, 0 diálogos completamente correctos de 10 y 6 fallos críticos. La revisión cruzada obtuvo 7 completas, 7 parciales y 26 fallidas, con 8 fallos críticos. Son revisiones de IA del proyecto, no evaluación humana independiente. El criterio sigue siendo 36/40, 8/10 y ningún fallo crítico o técnico, con una evaluación nueva reservada después de seleccionar un candidato.

El [estado actual y sus fuentes](../../docs/ESTADO_ACTUAL.md) registra los experimentos posteriores: original128 obtuvo 8 completas, pero perdió cuatro aciertos anteriores de C7 y tuvo siete críticas; relacional128 obtuvo seis completas y nueve críticas. Ambos quedaron en 0/10 diálogos completos y no se promovieron. No hay entrenamiento activo comprobado en este entorno; los presupuestos anteriores de 24 horas y dos horas están cerrados.

Desde la raíz del repositorio, con las dependencias de conversación instaladas:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

Escribe `/salir` para terminar y `/reiniciar` para borrar el historial de esa sesión. La interfaz carga los pesos locales mediante `ConversationEngine` y conserva las respuestas que realmente genera. No usa una API generadora ni respuestas prefabricadas.

La consola a 160 tokens se comprobó con cuatro turnos y reprodujo sus respuestas evaluadas. Los 40 turnos archivados en `evidence/` usaron un máximo de 32; la comprobación posterior a 160 produjo exactamente los mismos textos y tokens de C7 en esos casos, todos terminados por EOS. Esa equivalencia se limita a C7 y esos 40 casos; no se extiende automáticamente a otros candidatos.

Modelo: `unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2`. Contiene 6.983.937 parámetros, contexto de 512 tokens, vocabulario BPE propio de 8.192 tokens y cabeza de copia aprendida v1. Pesos, configuración y tokenizador están identificados por SHA-256 en `manifest.json`.

[Las 10 conversaciones completas](evidence/dialogues-actual.txt), [las respuestas originales en JSON](evidence/dialogues-actual.json) y las revisiones están en `evidence/`. Persisten errores al seguir nombres, relaciones y correcciones; también produce fragmentos y repeticiones. El entrenamiento posterior E1 obtuvo 3/40 y 0/10, por lo que no sustituyó esta referencia. Las versiones anteriores permanecen disponibles.

El [paquete de pesos y estados de la comparación relacional](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012) conserva también C7 y los candidatos descartados. Publicarlo no cambia su estado experimental ni inicia entrenamiento. Los cuatro archivos del modelo se conservan intactos.
