# Preparación del cierre CONTROL matched-v4

[prepare_control_final_delivery.py](prepare_control_final_delivery.py) usa sólo
biblioteca estándar. Su operación predeterminada es una comprobación de lectura:

```bash
python3 docs/learning-repair-20261013/prepare_control_final_delivery.py preflight
```

No crea archivos de entrega mientras el control sigue activo. La operación
`deliver` requiere un trabajador ausente, estado nativo terminal en el paso 4.096,
parada cooperativa `absolute_step_boundary`, recibo completo del runner y hashes
del código congelado coincidentes. Antes de escribir comprueba también los
checkpoints, el artefacto seleccionado y la exposición TRAIN exacta.

Sólo tras comprobar el cierre:

```bash
timeout 120s python3 docs/learning-repair-20261013/prepare_control_final_delivery.py deliver
```

La entrega preserva LAST en partes de hasta 32 MiB bajo
`checkpoints/learning-repair-20261013/natural-latam-v4/control-final/LAST/`.
Verifica los hashes de cada parte y los archivos completos recorriendo las partes
en orden. Conserva estado, Adam, RNG, scheduler, orden y nombres nativos.

Si BEST permanece en el paso cero y sus pesos coinciden con SFT072 BEST4.096,
guarda los cuatro archivos nativos de metadatos y una referencia verificada a los
pesos propios ya publicados. Su exposición nueva CONTROL es cero. El optimizador
inicial del control se conserva; los pesos del padre no se presentan como un
nuevo aprendizaje. La referencia de inferencia tampoco se carga directamente
con el motor: debe resolverse `artifactDirectory`.

Para reponer BEST desde su referencia, después de la entrega y con un destino
nuevo que conserve el nombre original:

```bash
python3 docs/learning-repair-20261013/prepare_control_final_delivery.py restore-best-reference \
  --reference-dir checkpoints/learning-repair-20261013/natural-latam-v4/control-final/BEST-reference/NOMBRE_BEST_NATIVO \
  --destination private/control-final-recovery/checkpoints/NOMBRE_BEST_NATIVO
```

El directorio padre del destino debe existir. LAST se recompone con
`examples/package_native_checkpoint.py restore`, conservando asimismo su nombre.
Ambos comandos reponen bytes y no arrancan entrenamiento; no se han ejecutado.
Una recuperación del entrenador requiere además el corpus y código congelados,
y conserva el límite ya alcanzado de 4.096 pasos.

Si el cierre selecciona un BEST aprendido con pesos distintos, el helper puede
empaquetarlo como otro checkpoint nativo. Su presupuesto predeterminado de nuevos
archivos es 256 MiB; `--max-new-disk-mib 512` permite dos paquetes completos.
El cálculo rechaza una entrega que exceda el límite antes de publicarla.

Las operaciones reducen su afinidad a una CPU, usan prioridad `nice +10`, limitan
el espacio de direcciones a 512 MiB y CPU a 120 segundos. El comando `timeout`
limita también el tiempo de pared. No importan Torch, no leen evaluaciones
reservadas ni cachés VALIDATION/TEST, no cargan modelos o pickle y no envían señales
al trabajador ni al supervisor. La exportación de inferencia LAST y la revisión
conversacional se realizarán por separado.
