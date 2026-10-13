# Checkpoints finales CONTROL matched-v4

[LAST](LAST/step-000004096-b6bdfcddec62449b82825fd7b9f838e5/) contiene los cinco
archivos nativos en nueve partes de hasta 32 MiB, con hashes de partes y archivos
completos verificados. Conserva las 4.096 actualizaciones y su optimizador/RNG.

[BEST-reference](BEST-reference/step-000000000-b0277331e4ee4dc0b7c185082d36571b/)
conserva los cuatro archivos nativos de metadatos del paso cero y una referencia
verificada a los pesos existentes del padre. Se recompone con la operación
`restore-best-reference` del helper; el directorio no se carga directamente.
Su optimizador inicial es propio de CONTROL y su nueva exposición es cero.

La [entrega final](../../../../docs/learning-repair-20261013/natural-latam-v4-control-final-delivery.md)
conserva instrucciones exactas para reponer ambos checkpoints, el corpus y código
requeridos y los límites que el estado mantiene. No se ejecutó restauración ni
entrenamiento durante esta entrega.
