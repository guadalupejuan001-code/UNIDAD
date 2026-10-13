# Probar Unidad y conectarla con Hormiga X

Revisión del 12 de octubre de 2026. Unidad se puede probar con su propio motor.
El importador actual de Hormiga X **no admite directamente este artefacto**.
Esta revisión inspecciona código; no acredita una prueba en un teléfono ni una
integración implementada.

| Aspecto | Hormiga X actual | Unidad evaluada |
| --- | --- | --- |
| Importación local | Un archivo GGUF; verifica su cabecera y SHA-256. | Carpeta con `weights.safetensors`, `tokenizer.json`, `config.json` y `manifest.json`. |
| Motor | llama.cpp por JNI, revisión fijada a `74d4f5b041ad837153b0e90fc864b8290e01d8d5`. | Transformer propio con cabeza de copia, ejecutado por `ConversationEngine` en PyTorch. |
| HTTP | Formato OpenAI compatible, `chat/completions`. | Contrato propio, `POST /v1/requests`, operación `conversation.reply`. |
| Contexto y salida | El razonamiento solicita hasta 1.800 tokens de salida, además del contexto de tarea. | Contexto total de 512 tokens; evaluación con hasta 32 tokens de salida. |
| Uso cognitivo | Solicita JSON con esquema, evidencias y herramientas; valida la respuesta. | La capacidad de razonamiento estructurado y uso de herramientas no está validada. |
| Privacidad | El proveedor remoto rechaza los contextos privados. | El motor puede ejecutarse localmente; un puente HTTP no equivale por sí solo a un proveedor local de Hormiga. |

Cambiar el nombre del archivo a `.gguf` no convierte arquitectura, pesos ni
tokenizador. Hace falta implementar y comprobar soporte para esta arquitectura
en un motor admitido, o incorporar un proveedor local que ejecute el motor de
Unidad y adapte su importación. Esa integración debe conservar identificación
de los pesos, cancelación, límites de contexto y reglas de privacidad.

Un puente de protocolo HTTP es otra posibilidad para una prueba controlada,
pero los endpoints actuales no son intercambiables. Tampoco basta para convertir
las respuestas conversacionales de Unidad en el razonamiento estructurado que
necesitan las Hormigas. El adaptador debe declarar capacidades reales y fallar
con claridad ante las que no estén implementadas. No debe enviar información
privada a un remoto mediante una reclasificación artificial.

## Prueba disponible ahora

Con el código de Unidad y sus dependencias conversacionales instaladas, cargar
la carpeta exacta del candidato indicada en el informe de resultados:

```bash
python -m ia_propia conversation converse --model /ruta/al/candidato \
  --temperature 0 --seed 42 --max-new-tokens 32
```

`/reiniciar` limpia el diálogo y `/salir` termina. Las respuestas provienen de
los pesos elegidos. El identificador del modelo debe coincidir con el informe;
no se sustituye silenciosamente por Qwen u otro modelo. Una prueba de carga
correcta no demuestra todavía conversación competente.

La consola existente de Hormiga X es una consola de comandos de su sistema,
no un chat libre. Por eso, incluso con un proveedor compatible, conversar con
Unidad desde esa interfaz necesitaría un flujo explícito de prueba. Esta tarea
no cambió la aplicación ni creó un APK nuevo.

## Fuentes inspeccionadas

Hormiga X, commit `1751ce4fe5daed3bea998ecb26b438dce491b20b`:

- [Importador, motor local y proveedor remoto](https://github.com/guadalupejuan001-code/Hormiga-X/blob/1751ce4fe5daed3bea998ecb26b438dce491b20b/core/src/main/kotlin/hormigax/core/Providers.kt).
- [Contrato de razonamiento y presupuesto de salida](https://github.com/guadalupejuan001-code/Hormiga-X/blob/1751ce4fe5daed3bea998ecb26b438dce491b20b/core/src/main/kotlin/hormigax/core/Reasoning.kt).
- [Motor JNI](https://github.com/guadalupejuan001-code/Hormiga-X/blob/1751ce4fe5daed3bea998ecb26b438dce491b20b/native/README.md).
- [Interfaz y consola Android](https://github.com/guadalupejuan001-code/Hormiga-X/blob/1751ce4fe5daed3bea998ecb26b438dce491b20b/android/README.md).

En Unidad: [motor conversacional](../src/ia_propia/conversation/runtime.py),
[HTTP](../src/ia_propia/brain/http.py) y
[CLI conversacional](../src/ia_propia/conversation/__main__.py).
