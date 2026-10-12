# Pesos propios de Unidad: paquete experimental

Aquí están los pesos, configuraciones, tokenizadores, ocho checkpoints completos, corpus y evaluaciones del contraste relacional. C7 sigue como referencia; ninguno de los candidatos alcanza conversación básica. Este paquete no inicia entrenamiento ni cambia los pesos predeterminados.

GitHub admite cada archivo de esta carpeta; el ZIP de 339.107.857 bytes se dividió en seis partes consecutivas de hasta 64 MiB. Son bytes del archivo original, sin volver a comprimir, editar ni sustituir pesos. SHA-256 del ZIP reconstruido:

`5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`

Clona esta rama para obtener las seis partes y el proyecto:

```bash
git clone --single-branch --branch artifacts/unidad-contraste-relacional-20261012 https://github.com/guadalupejuan001-code/UNIDAD.git unidad-con-pesos
cd unidad-con-pesos
python weights-packages/unidad-contraste-relacional-20261012/reconstruir.py
```

El script verifica cada parte y el ZIP final. No sobrescribe un archivo existente. El recibo adjunto identifica C7 y ambos candidatos propios, los 370 archivos restaurados, 290 objetos únicos y ocho snapshots.

Después extrae `restaurar.py` del ZIP, lee las instrucciones incluidas y usa un directorio nuevo:

```bash
python restaurar.py Unidad-Contraste-Relacional-Experimental.zip unidad-restaurada
cd unidad-restaurada
python -m pip install -e '.[conversation]'
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

La rama contiene el mismo código de la entrega `57a8190` del PR #3, más el paquete. Código del experimento: `8b466cc`. Main, el PR #3, las otras ramas y los archivos históricos se conservan. Los presupuestos anteriores están cerrados; conservar estados no autoriza ampliar automáticamente los 128 pasos completados. Este paquete tiene nombre propio y no sustituye el ZIP diagnóstico ausente aportado por el usuario.

## Estado actual e historial del paquete

Consulta [el estado actual del proyecto](../../docs/ESTADO_ACTUAL.md). El ZIP conserva el snapshot verificado de su creación; las actualizaciones posteriores de documentación no reescriben sus bytes. Los hashes del manifiesto de partes y del archivo original se mantienen. C7 ya tenía pesos de inferencia en main; este paquete añade los candidatos y estados del contraste relacional.
