# Estado nativo final de la base propia

La etapa `foundation-15m` terminó por `total_time_budget` en step **8.914**.
Se preservan sus dos checkpoints cerrados completos. Aunque ambos tienen el
mismo número de update, sus nombres y estados nativos difieren; no se fusionan
ni se cambia su historia.

| Uso nativo | Directorio | Bytes originales | Partes |
| --- | --- | ---: | ---: |
| BEST, seleccionado por NLL de validation | `step-000008914-ec5f112fedf545ff9fd3a0bafeeb2365` | 182.406.109 | 9 |
| LAST, último estado para reanudar | `step-000008914-fbd42a9742a243efa27110956f99aaa7` | 182.406.109 | 9 |

Cada directorio contiene partes de como máximo 32 MiB y sus dos manifiestos.
Los cinco archivos nativos originales, incluidos Adam/scheduler/RNG y estado,
se copiaron con hashes idénticos antes y después; las partes se verificaron.
El [recibo final](../../../../docs/learning-repair-20261013/foundation-15m-final-delivery.json)
conserva las identidades. Los hashes de los manifiestos de paquete son:

- BEST: `219c2af67b0d39b2a2e24e53d8610ec71c0d1fb85e4fd8bbb36c88443728659b`.
- LAST: `92fc7ecfe9f207c4a5c17a3d9313d8b15eb9ca5bda791e21142249431b32fbba`.

La [contabilidad final](../../../../docs/learning-repair-20261013/foundation-15m-final-target-exposure.json)
registra 22.906.669 posiciones objetivo en 71.297 ventanas. Son posiciones
procesadas, no palabras distintas ni una aceptación conversacional. La base
permanece experimental y C7 se conserva.

Para reponer los archivos, usar
[package_native_checkpoint.py](../../../../examples/package_native_checkpoint.py)
con `restore`, en destinos nuevos que mantengan los nombres originales. Reponer
ambos checkpoints bajo el mismo `checkpoints/` conserva los punteros BEST y
LAST. El procedimiento y las condiciones de reanudación están en
[checkpoint-packaging.md](../../../../docs/learning-repair-20261013/checkpoint-packaging.md).
La restauración no se ejecutó durante esta entrega y no arranca entrenamiento.
