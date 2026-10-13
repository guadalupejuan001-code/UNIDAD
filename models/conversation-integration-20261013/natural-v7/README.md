# Unidad: candidato experimental natural-v7

Estos pesos propios no han superado la aceptación de conversación: 36/40 respuestas completas, 8/10 diálogos completos y ningún fallo crítico o técnico. No sustituyen C7 ni acreditan conversación competente.

Modelo: `unidad-conversation-7387a514e1be02fa6bb2fa7b14d113edb48adf0e34b21a1be94366c762f37fd1`. Padre: `unidad-conversation-9081da24a3a73e748887dd66ee937798ef6808579ed5cdb78c3b27b70639cd9d`. Se entrenaron 512 pasos; se publica la selección por validación del paso **128**, con NLL 4.71816483. La pérdida no determina calidad conversacional. Los cuatro archivos nativos se copiaron sin modificar sus bytes.

Desde la raíz del repositorio:

```bash
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-integration-20261013/natural-v7 --max-new-tokens 96 --temperature 0 --seed 42
```

[Recibos y comprobación](../../../docs/conversation-integration-20261013/training/natural-v7/verification.json) conservan configuración, linaje propio iniciado desde pesos aleatorios, corpus, parámetros realmente modificados y hashes. Las rutas `/workspace/UNIDAD/private/...` de los recibos son rutas locales históricas. La evaluación conversacional se conserva aparte en el informe de integración; estos resultados de entrenamiento no reemplazan sus respuestas.

La atribución lingüística heredada se conserva en [spanish-foundation-v1](../../spanish-foundation-v1/README.md) y [conversation-language-v7](../../conversation-language-v7/README.md): OASST declara Apache-2.0; Wikipedia conserva sus distintas declaraciones CC BY-SA/GFDL. Los diálogos nuevos ficticios fueron redactados por asistentes de IA del proyecto. Estas referencias no certifican una licencia específica de distribución de los pesos.
