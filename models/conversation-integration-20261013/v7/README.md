# Unidad: candidato experimental v7

Estos pesos propios no han superado la aceptación de conversación: 36/40 respuestas completas, 8/10 diálogos completos y ningún fallo crítico o técnico. No sustituyen C7 ni acreditan conversación competente.

Modelo: `unidad-conversation-9d27cc9a24a38beb2e6139dc62dcb13850ac4ac068ed89e6ce65eeef98bc7bf5`. Padre: `unidad-conversation-9081da24a3a73e748887dd66ee937798ef6808579ed5cdb78c3b27b70639cd9d`. Se entrenaron 256 pasos; se publica la selección por validación del paso **256**, con NLL 4.63546030. La pérdida no determina calidad conversacional. Los cuatro archivos nativos se copiaron sin modificar sus bytes.

Desde la raíz del repositorio:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-integration-20261013/v7 --max-new-tokens 96 --temperature 0 --seed 42
```

[Recibos y comprobación](../../../docs/conversation-integration-20261013/training/v7/verification.json) conservan configuración, linaje propio iniciado desde pesos aleatorios, corpus, parámetros realmente modificados y hashes. Las rutas `/workspace/UNIDAD/private/...` de los recibos son rutas locales históricas. La evaluación conversacional se conserva aparte en el informe de integración; estos resultados de entrenamiento no reemplazan sus respuestas.

La atribución lingüística heredada se conserva en [spanish-foundation-v1](../../spanish-foundation-v1/README.md) y [conversation-language-v7](../../conversation-language-v7/README.md): OASST declara Apache-2.0; Wikipedia conserva sus distintas declaraciones CC BY-SA/GFDL. Los diálogos nuevos ficticios fueron redactados por asistentes de IA del proyecto. Estas referencias no certifican una licencia específica de distribución de los pesos.
