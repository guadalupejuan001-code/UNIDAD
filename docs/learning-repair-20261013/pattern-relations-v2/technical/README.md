# Paquete técnico de pattern-relations-v2

La etapa propia terminó en **512 actualizaciones**, tras
**72,28323967199685 segundos**, con motivo nativo `signal` y parada cooperativa
`absolute_step_boundary`. Seleccionó BEST en el paso **128**, modelo propio
`unidad-conversation-af515755f6eb176ed9968235231f24d8abdebca002f881a327ba69651bed1591`,
con NLL de validación **2,004439949989319**. El padre es el modelo propio C0ee
[reference-wiki1](../../../../models/learning-repair-20261013/pattern-relations-v1/reference-wiki1/).

El [candidato de inferencia existente](../../../../models/learning-repair-20261013/pattern-relations-v1/candidate-best/)
conserva exactamente los cuatro archivos del snapshot BEST seleccionado. Esta
entrega los verificó sin copiarlos, sobrescribirlos o ejecutar inferencia.

El [recibo técnico](delivery-receipt.json) identifica los **15.769.344 parámetros**,
todos entrenables y ninguno congelado, propietario del runner, configuración,
recursos y procedencia. Se copiaron byte por byte el diseño, estado final, recibo
del runner, log nativo, estado/manifiesto BEST y manifiesto de caché. La
[configuración](training-config.json) se extrajo de su diseño y coincide con ambos
estados nativos. El [manifiesto de entrega](MANIFEST.json) registra sus archivos.

## Recuperabilidad comprobada

El [paquete BEST128](../../../../models/learning-repair-20261013/pattern-relations-v1/native-best128-parts/)
conserva **189.387.541 bytes nativos** en nueve partes de hasta 32 MiB. Su
manifiesto SHA256 es
`e36a3577315de97c6114c6b5755400d9e546a3b5b8fd3ae33c787855d393b654`.

Se ejecutó una restauración real con el helper de biblioteca estándar y se
comprobaron los tamaños y SHA256 de los cinco archivos contra el BEST original.
No se importó Torch, interpretó el pickle de `resume.pt` ni cargó un modelo. Se
eliminó exclusivamente el directorio temporal creado para esa comprobación.

Para reponer los bytes nuevamente, desde la raíz del repositorio y con un destino
nuevo que conserve el nombre nativo:

```bash
mkdir -p private/pattern-best128-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  models/learning-repair-20261013/pattern-relations-v1/native-best128-parts \
  private/pattern-best128-recovery/checkpoints/step-000000128-899fe8371706440faa275a0cdd4a5cfc
```

Esta instrucción recompone archivos y no arranca entrenamiento. Conserva Adam,
RNG, scheduler, orden, pasos y tiempo del BEST128 original. El estado de la
corrida completa al cierre corresponde a **LAST512**, que permanece intacto en
la ruta privada original. Sus metadatos y hashes se respaldaron adicionalmente
en `private/learning-repair-20261013/delivery/pattern-relations-v2-last512/`;
no se añadieron copias grandes de sus pesos u optimizador ni un paquete público
LAST512. El recibo identifica sus cinco archivos y ese alcance.

Las referencias históricas a rutas privadas o bootstrap conservadas en metadatos
son procedencia. Esta entrega no las ejecutó ni verificó una reanudación completa
del runtime. La etapa fue un ajuste desde pesos propios, con optimizador y
scheduler nuevos (`resume=false`), y la restauración preserva sus contadores.
Una recuperación del entrenador requiere el corpus exacto, código y diseño
congelados y una selección explícita del estado que se recupera.

## Exposición real

El [ledger TRAIN](target-exposure.json) cuenta etiquetas supervisadas con
`label >= 0`, excluye `-100` e incluye EOS. Los barajados y el RNG completo
coinciden con su reproducción mediante semilla 20261013. El lote lógico es cuatro
y el microbatch máximo dos.

| Exposición de esta etapa | BEST128 | LAST512 |
| --- | ---: | ---: |
| Ventanas consumidas | 512 | 2.048 |
| Tokens de entrada sin padding | 71.970 | 287.639 |
| Posiciones supervisadas, incluido EOS | 4.302 | 17.178 |
| Posiciones EOS | 512 | 2.048 |
| Posiciones supervisadas sin EOS | 3.790 | 15.130 |

Se repitieron **48 ventanas TRAIN**: BEST completó diez épocas y LAST cuarenta y
dos, ambos con cursor 32 en la siguiente. Estas cifras no acreditan
generalización. TEST está excluido; no se leyeron corpus, cachés VALIDATION/TEST
ni evaluaciones reservadas durante la entrega. Los registros públicos corresponden
al experimento de ficción autorizado.

La preparación usó una CPU, prioridad nice19, hasta 512 MiB de espacio de
direcciones y 120 segundos por operación. No entrenó, evaluó respuestas, alteró
fuentes/pesos originales, modificó documentación principal o ejecutó operaciones
Git. La revisión es de IA del proyecto; la aceptación conversacional permanece
en falso. La subida a GitHub se verifica por separado.
