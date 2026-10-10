# Especialista opcional de atención

Este componente es una capacidad inicial reutilizable de IA Propia. No define
las categorías de trabajo del futuro asistente ni determina qué puede pedirle
el usuario. Entrena una red pequeña desde pesos aleatorios para anticipar el
cambio de una métrica no negativa de una señal. El primer ejemplo usa visitas
Wikimedia; no interpreta comportamiento, genera lenguaje o ejecuta tareas por sí
solo. El proyecto puede incorporar otras capacidades sin convertir esta tarea en
su identidad.

## Usar el paquete independiente

Con el proyecto instalado, el módulo funciona sin Hormiga-X:

```bash
python -m ia_propia.trend --help
python -m ia_propia.trend predict \
  --model models/attention-v1.json \
  --input examples/attention-observations.json \
  --as-of 2024-10-16T00:00:00Z --historical
```

El archivo guardado conserva su contrato original y sus pesos. `--historical`
declara una reproducción retrospectiva con evidencia recuperada después del
reloj virtual; la salida incluye `historicalReplay=true`,
`liveForecastValidated=false` y una advertencia. No representa un pronóstico
realizado en 2024. Sin ese flag se exige que capturas, recuperación y entrenamiento
existieran en el instante pedido y que el horizonte todavía sea futuro.
En ambos modos se rechazan ventanas cuyo origen preceda al último resultado
usado para entrenar o elegir el modelo (`training.fitCutoffAt`).

Para entrenar un experimento nuevo sin sustituir el artefacto conservado:

```bash
python -m ia_propia.trend train \
  --input examples/attention-observations.json \
  --source wikimedia:en.wikipedia.org --metric views \
  --output models/attention-experiment.json \
  --report docs/attention-experiment.json
```

La configuración predeterminada usa ventana de cuatro capturas, horizonte de un
paso, cadencia diaria exacta (86400000 ms), ocho neuronas ocultas, semilla 42,
Adam con tasa 0,01, máximo de 2000 épocas y paciencia de 150. Se configura mediante
`--window`, `--horizon-steps`, `--cadence-ms`, `--hidden-size`, `--seed`,
`--learning-rate`, `--epochs` y `--patience`. El informe registra las versiones
de Python y NumPy. No se descargan pesos preentrenados.

## Aprendizaje, datos y contrato

La entrada acepta una lista de observaciones con ID, señal, fecha medida, texto,
métricas, procedencia y privacidad PUBLIC. Por compatibilidad también lee un
estado JSON de esquema 1 con mapas de observaciones y señales; las señales usadas
deben ser explícitamente PUBLIC. Se rechazan capturas privadas o sin clasificación,
procedencia incompleta, colisiones de IDs, timestamps ambiguos, claves JSON
duplicadas, valores no finitos y fechas futuras. Las copias idénticas de un ID
se deduplican con conteo registrado. Cada experimento usa una sola fuente y
métrica; una captura privada de la misma señal eleva su sensibilidad.

Las características son los tres deltas sucesivos de `log1p(valor)` y el último
nivel logarítmico para la ventana de cuatro capturas. La ventana y el horizonte
deben estar completos con la cadencia declarada; no se imputa un hueco. El objetivo
es `log1p(valor futuro) - log1p(último valor)`. Una capa `tanh` y una salida lineal
aprenden ese cambio con backpropagation y Adam implementados en NumPy. El valor
predicho se obtiene con `max(0, expm1(log1p(último valor) + salida))`.
Los desbordamientos numéricos se rechazan.

Los campos heredados `hormiga-trend-mlp` y el prefijo `hormiga-trend-` en `modelId`
se mantienen como formato de intercambio, con normalización, pesos, configuración,
procedencia y evaluación completas. No son una importación del programa anterior.
El [registro de procedencia](provenance.md) identifica el commit y los hashes.

## Resultado conservado y verificación

Las 30 capturas públicas de dos señales producen 22 ventanas: 12 train, 4
validation y 6 test. Los splits son globales y cronológicos por fecha del target;
las señales de la misma fecha permanecen juntas. Para horizontes largos se purgan
etiquetas posteriores al primer origen del split siguiente. La normalización usa
sólo train y sólo validation elige el checkpoint. El conjunto test no selecciona
pesos ni hiperparámetros.

El [informe original](attention-training-v1.json) eligió la época 38. El RMSE de
crecimiento logarítmico de test fue **0,178261**, frente a **0,133981** de
persistencia, que conserva el último valor. Este modelo **todavía rinde peor que
ese baseline**. Se mantiene el resultado íntegro y no se ajusta usando test.

Las pruebas extraídas comprueban aprendizaje real sobre una serie funcional,
independencia del escalador y pesos respecto al test, causalidad temporal,
privacidad, rechazo de overflow, identidad del artefacto, roundtrip y el recorrido
CLI con observaciones reales. Para ejecutarlas desde el árbol fuente:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_trend.py -v
```

El fixture numérico compartido vive en `tests/fixtures/parity.json`. Las pruebas
técnicas no prueban inteligencia cultural, comprensión, autonomía ni calidad de
un asistente digital. El artefacto sigue marcado como experimental y sin
aceptación de producción; no ofrece probabilidades ni confianza calibrada.
