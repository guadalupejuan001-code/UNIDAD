# Entrenamiento de SOVENGI: memoria recuperada y conversación

Se inicia una etapa finita desde el modelo propio que carga SOVENGI en `main`
`2d989a5`. La hipótesis es concreta: ajustar respuestas a hechos recuperados y a
correcciones, manteniendo repaso conversacional. La integración ya recuperaba
la nota correcta, pero el modelo podía contestar con texto incorrecto.

## Diseño fijado antes de entrenar

- Padre: `reference-wiki1`, 15.769.344 parámetros; identidad y SHA-256 fijados en
  [la configuración](../../configs/sovengi-memory-stage.json).
- Nueva etapa con pesos y tokenizador propios; optimizador nuevo. No es una
  reanudación del optimizador de otra corrida, ni sustituye un entrenador activo.
- 320 ejemplos ficticios de enseñanza y 32 de validación, separados por familia
  de entidades; pares con la misma pregunta y hechos distintos. Se añade una
  muestra de al menos 160/24 registros de conversación, conservando grupos,
  mensajes, procedencia y particiones originales.
- Desarrollo: 20 consultas de memoria de entidades no entrenadas y cinco turnos
  adicionales de retención. La corrección usa la respuesta realmente generada
  como historia. Las respuestas esperadas sólo se usan al puntuar.
- Hasta 512 actualizaciones, 900 segundos de entrenamiento y dos hilos de CPU;
  el trabajo completo de Actions tiene un límite de 30 minutos. Se conservan
  checkpoints completos y se selecciona por validación, sin consultar TEST.
- Mismo `Sovengi` y `ConversationEngine` antes/después: Torch, caché habilitada,
  temperatura cero y máximo 64 tokens. La memoria usa el contrato del producto.
- Ningún mensaje personal se convierte en corpus. Los ejemplos ficticios están
  identificados como escritos por Codex; no son respuestas programadas durante
  la inferencia.

El workflow publica el candidato, corpus congelado y respuestas en su rama,
y guarda los checkpoints reanudables como artefacto de Actions durante 90 días.
No cambia automáticamente `configs/gnosis.json`. El recuento de texto exacto
es un diagnóstico estricto, no una calificación semántica ni la aceptación
conversacional de 36/40 respuestas y 8/10 diálogos. El cierre debe registrar los
pasos reales, resultados y decisión; disponer de este diseño no significa que
la corrida haya terminado.

## Ramas revisadas

La [auditoría de ramas](branch-audit.json) fija la comprobación anterior a esta
etapa: cero PR abiertos, 13 ramas ya contenidas en `main` y una rama histórica
`artifacts/unidad-contraste-relacional-20261012` con dos commits propios.
Sus cambios únicos son un paquete de 339 MB con pesos y ocho checkpoints del
12 de octubre. No contiene código de ejecución pendiente. Parte de su
documentación ya está idéntica en `main`; el resto tiene versiones posteriores.

Se conserva el [archivo histórico en su rama original](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012),
con SHA-256 del ZIP `5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`.
No se duplican esos 339 MB en el árbol actual ni se revierte documentación para
hacer desaparecer una rama del listado. Por tanto, existe una rama de archivo
fuera de `main`; no se afirma falsamente que todas sus revisiones estén fusionadas.

## Reproducir

Desde un checkout limpio del commit de preparación, con dependencias
conversacionales y Node.js instalados:

```bash
python -u examples/train_sovengi_memory.py
```

La herramienta rechaza sobrescribir el corpus congelado y verifica el padre y
el corpus de repaso antes de generar o entrenar. Para reanudar un checkpoint,
se utiliza el entrenador común con el corpus original y `resume=True`; no se
vuelve a ejecutar la preparación sobre una corrida existente.
