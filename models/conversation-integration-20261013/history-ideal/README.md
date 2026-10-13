# Unidad: candidato experimental history-ideal

Estos pesos propios no han superado la aceptación de conversación: 36/40 respuestas completas, 8/10 diálogos completos y ningún fallo crítico o técnico. No sustituyen C7 ni acreditan conversación competente.

Modelo: `unidad-conversation-42e2f6c98ccb35d9f6585d73ed8c7c536f2b355745dc2022013d6b7abf80353b`. Padre: `unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2`. Se entrenaron 512 pasos; se publica la selección por validación del paso **512**, con NLL 0.06161023. La pérdida no determina calidad conversacional. Los cuatro archivos nativos se copiaron sin modificar sus bytes.

Desde la raíz del repositorio:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-integration-20261013/history-ideal --max-new-tokens 96 --temperature 0 --seed 42
```

[Recibos y comprobación](../../../docs/conversation-integration-20261013/training/history-ideal/verification.json) conservan configuración, linaje propio iniciado desde pesos aleatorios, corpus, parámetros realmente modificados y hashes. Las rutas `/workspace/UNIDAD/private/...` de los recibos son rutas locales históricas. La evaluación conversacional se conserva aparte en el informe de integración; estos resultados de entrenamiento no reemplazan sus respuestas.

La atribución lingüística heredada se conserva en [spanish-foundation-v1](../../spanish-foundation-v1/README.md) y [conversation-language-v7](../../conversation-language-v7/README.md): OASST declara Apache-2.0; Wikipedia conserva sus distintas declaraciones CC BY-SA/GFDL. Los diálogos nuevos ficticios fueron redactados por asistentes de IA del proyecto. Estas referencias no certifican una licencia específica de distribución de los pesos.
