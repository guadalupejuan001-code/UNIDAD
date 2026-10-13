# Estados nativos del ajuste conversacional propio

`intermediate-step004096/step-000004096-37b0b60acf4546d398078b3c407e600d/`
conserva el checkpoint BEST inmutable de la primera pausa natural de SFT. Sus
cinco archivos originales contienen pesos, tokenizador, estado, manifiesto y
Adam/scheduler/RNG; se distribuyen en nueve partes de como máximo **32 MiB**.
No se copió LAST durante la reanudación ni se alteró el entrenamiento activo.

La identidad del manifiesto de transporte es
`7e59b058e3e0405557daeaa4fcac0fcca0647824802872cc43f21823208c8227`.
Los hashes de los cinco archivos nativos coincidieron antes y después del
empaquetado, y se comprobaron los hashes de todas las partes publicadas. El
[recibo de entrega](../../../docs/learning-repair-20261013/conversation-15m-intermediate-step004096-delivery.json)
conserva tamaños y hashes; no cargó Torch ni interpretó `resume.pt`.

```bash
python examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/conversation-15m/intermediate-step004096/step-000004096-37b0b60acf4546d398078b3c407e600d \
  /tmp/unidad-sft-step4096-restaurado
```

La salida debe ser una ruta nueva. Esta entrega no ejecutó la restauración;
distribuye las partes verificadas y su instrucción reproducible. Para reconstruir
una carpeta de corrida nueva se mantienen las condiciones descritas en
[checkpoint-packaging.md](../../../docs/learning-repair-20261013/checkpoint-packaging.md):
corpus idéntico, configuración idéntica y registro explícito de que la corrida
restaurada está detenida antes de reanudar. No se apunta ese procedimiento a la
corrida actualmente activa.

La primera invocación se detuvo naturalmente en el paso 5.073 tras
3.600,451 segundos. El mismo supervisor lanzó `--resume` desde LAST a las
07:17:08 UTC, conservando el reloj acumulado y los límites de la etapa: hasta
8.192 pasos y umbral de 7.200 segundos. BEST es la selección para inferencia;
LAST es el punto usado para continuar el aprendizaje. El
[recibo de reanudación](../../../docs/learning-repair-20261013/conversation-15m-first-resume.json)
conserva la evidencia de procesos y progreso sin afirmar una comparación
independiente tensor por tensor. `optimizerReset=true` en el estado histórico
describe la inicialización de la etapa SFT desde su base propia; no un reinicio
del optimizador al reanudar esta misma corrida.

Este respaldo intermedio y la menor pérdida de validación no acreditan
conversación competente ni reemplazan la referencia C7.
