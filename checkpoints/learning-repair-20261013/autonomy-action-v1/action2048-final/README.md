# Checkpoints posteriores al ajuste ACTION 2.048

[LAST](LAST/step-000000256-ecc22755d8084d4994ff315534a0b1d1/) contiene los cinco
archivos nativos en nueve partes de hasta 32 MiB, con hashes de partes y archivos
completos comprobados.

[BEST-reference](BEST-reference/step-000000256-df25991a5d06418fb5970aa846ae3eda/)
conserva sus propios `state.json` y `manifest.json` y referencia los pesos,
Adam/RNG y tokenizador idénticos de LAST. Sus cinco archivos se recomponen con
`docs/learning-repair-20261013/restore_action_native_reference.py`; el directorio
de referencia no se carga directamente como un checkpoint.

La [entrega final](../../../../docs/learning-repair-20261013/autonomy-action2048-final-delivery.md)
conserva comandos exactos y exposición real. Ambos checkpoints posteriores son
recuperables. El checkpoint nativo inicial fue eliminado por la retención y su
optimizador/RNG original no se encontró; se conservan el padre de inferencia y la
evidencia inicial disponible, sin presentarlos como el estado nativo perdido.

Estos archivos y la pérdida de validación no demuestran competencia operativa.
