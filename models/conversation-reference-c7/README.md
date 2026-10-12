# Unidad C7: referencia experimental

Estos son pesos propios de Unidad. **Todavía no cumplen la aceptación de conversación básica.** Se conservan porque superaron a los candidatos posteriores en la misma evaluación de desarrollo.

La revisión del proyecto obtuvo 7 respuestas completas de 40, 0 diálogos completamente correctos de 10 y 6 fallos críticos. La revisión cruzada obtuvo también 7 respuestas completas, con 8 fallos críticos. Son revisiones de agentes del proyecto, no evaluación humana independiente. El criterio sigue siendo 36/40, 8/10 y ningún fallo crítico.

Desde la raíz del repositorio, con las dependencias de conversación instaladas:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 32 --temperature 0 --seed 42
```

Escribe `/salir` para terminar y `/reiniciar` para borrar el historial de esa sesión. La interfaz carga los pesos locales mediante `ConversationEngine` y conserva las respuestas que realmente genera. No usa una API generadora ni respuestas prefabricadas.

Modelo: `unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2`. Contiene 6.983.937 parámetros, contexto de 512 tokens, vocabulario BPE propio de 8.192 tokens y cabeza de copia aprendida v1. Pesos, configuración y tokenizador están identificados por SHA-256 en `manifest.json`.

[Las 10 conversaciones completas](evidence/dialogues-actual.txt), [las respuestas originales en JSON](evidence/dialogues-actual.json) y las revisiones están en `evidence/`. Persisten errores al seguir nombres, relaciones y correcciones; también produce fragmentos y repeticiones. El entrenamiento posterior E1 obtuvo 3/40 y 0/10, por lo que no sustituyó esta referencia. Las versiones anteriores permanecen disponibles.
