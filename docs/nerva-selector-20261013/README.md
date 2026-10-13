# NERVA: resultado de rendimiento y compatibilidad — 13 de octubre de 2026

La ruta opcional `nerva-fast` quedó implementada y medida sobre código identificado.
En SOVENGI predeterminado redujo el tiempo de NERVA un 9,3 % (7,431 → 6,740 s
por diez consultas), pero Torch con caché fue más rápido: 4,417 s.
**Se recomienda conservar Torch con caché como predeterminado.** La optimización
del selector se valora por sus resultados completos; no cambia pesos ni acredita
mejor calidad conversacional. Los cambios permanecen en ramas independientes.

## Generación completa medida

Mediana de tres rondas; cada ronda contiene diez consultas conocidas. Unidad: segundos
**por conjunto de diez consultas**, con límite de 160 tokens nuevos por respuesta.
Incluye selección, generación neuronal, arranque/IPC/cierre de Node. Excluye carga
inicial del modelo, HTTP, recuperación de memoria y herramientas externas.

| Modelo | Torch sin caché | Torch con caché | NERVA original con caché | NERVA optimizada con caché | Mejora frente a NERVA original |
|---|---:|---:|---:|---:|---:|
| C7 histórico | 2,524 | 0,912 | 4,617 | 3,669 | 1,259× |
| SOVENGI predeterminado (reference-wiki1) | 27,356 | 4,417 | 7,431 | 6,740 | 1,103× |
| Candidato de memoria (separado) | 31,142 | 4,913 | 8,099 | 7,392 | 1,096× |

Las medianas por respuesta individual también se conservan en los JSON; no son
el mismo estadístico que dividir la mediana del conjunto entre diez. Las muestras
crudas permiten revisar variabilidad, número de tokens y terminación.

## Selector dentro de Node

Milisegundos por decisión, cinco rondas de 60 llamadas tras 20 calentamientos.
Entradas preconstruidas; estos valores excluyen Python, IPC y la red neuronal.

| Vocabulario | Temperatura | Intérprete | Especialización | Aceleración aislada |
|---|---:|---:|---:|---:|
| 2048 | 0 | 0,776 | 0,711 | 1,092× |
| 2048 | 0.7 | 1,174 | 1,102 | 1,065× |
| 8192 | 0 | 4,475 | 2,947 | 1,519× |
| 8192 | 0.7 | 6,373 | 4,879 | 1,306× |

## Sobrecoste entre Python y Node

Medianas en milisegundos por decisión. Son experimentos de componentes separados:
**no deben sumarse como una atribución exacta**. IPC incluye lectura JSON y selección
en Node; la selección completa añade la construcción de datos desde Torch.

| Vocabulario | Ruta | Tensor a lista | Codificación JSON | IPC con petición preparada | Selección completa |
|---|---|---:|---:|---:|---:|
| 2048 | Original | 0,045 | 1,084 | 1,561 | 2,500 |
| 2048 | Optimizada | 0,048 | 1,080 | 1,165 | 2,279 |
| 8192 | Original | 0,174 | 4,086 | 6,738 | 11,105 |
| 8192 | Optimizada | 0,173 | 4,072 | 5,194 | 8,966 |

Los costes de análisis del programa y JSON dentro de Node están separados en
[selector-kernel.json](selector-kernel.json). Arranque/primera decisión y cierre
del proceso están registrados en [selector-ipc.json](selector-ipc.json).

La especialización evita interpretar de nuevo el AST y exportar globales generales.
Mantiene la serialización del tensor para conservar exactamente sus cargos de
operaciones: el prototipo anterior la omitía. El JSON y el salto de proceso permanecen;
un ahorro en el selector no se convierte íntegramente en ahorro de una respuesta.

## Equivalencia, límites y modelos

- 468 respuestas medidas y conservadas. 117 pares NERVA original/optimizada
  idénticos en todos los campos de salida salvo el nombre del backend.
- Con selección voraz, todos los IDs de tokens coinciden también entre los cuatro recorridos.
- Con muestreo, se compararon tres consultas × tres rondas por modelo, temperatura 0,7,
  top-k 40, top-p 0,9 y penalización 1,1; presupuesto 64 tokens. Los dos NERVA coinciden.
  Torch usa otro RNG: no se presenta igualdad de muestras ni aceleración comparable si difieren.
- Hashes de configuración, tokenizador y pesos verificados antes y después: sin cambios.
- C7 se conserva como referencia histórica. `reference-wiki1` era y sigue siendo el
  predeterminado de SOVENGI al fijar el experimento. El candidato de memoria se midió
  aparte; no se promovió como modelo predeterminado.

- C7 histórico: 0/30 respuestas terminaron por el límite de tokens. Mismos cierres en las rutas equivalentes.
- SOVENGI predeterminado (reference-wiki1): 6/30 respuestas terminaron por el límite de tokens. Mismos cierres en las rutas equivalentes.
- Candidato de memoria (separado): 9/30 respuestas terminaron por el límite de tokens. Mismos cierres en las rutas equivalentes.

Terminar una llamada de generación no implica completar satisfactoriamente una
respuesta. No hubo entrenamiento, evaluación semántica nueva ni selección del
candidato por estas consultas; no se utilizaron respuestas esperadas para ajustar pesos.

## Compatibilidad y validación

Los siete archivos del runtime NERVA 0.2 de UNIDAD conservan sus hashes originales.
Mientras se trabajaba, el repositorio independiente avanzó a NERVA 0.3; se reconciliaron
esos cambios en la rama independiente sin trasladar el runtime nuevo a UNIDAD.
Las dos extensiones coinciden byte por byte entre repositorios y funcionan con 0.2.0
y 0.3.0. Se respetan los contextos de presupuesto y pasos adicionales de 0.3.

La nueva API recibe sólo los datos numéricos del programa fijo identificado por hash.
Mantiene el algoritmo f64, los límites de pasos/operaciones, 8 MiB, 1.024 decisiones,
timeout, prohibición de elegir código/rutas/herramientas desde peticiones y limpieza.
Rechaza getters sin invocarlos, símbolos, arrays dispersos y prototipos personalizados.
La API pública `runProgram`/`runSource2`, la CLI y los backends existentes se conservan.

Pruebas: 134 pruebas locales NERVA 0.3; CI en Node 18/22/24; 288 casos de equivalencia
del selector por versión (0.2 y 0.3), incluidos límites exactos de presupuesto. Las
pruebas Python pertinentes y la CI completa se identifican en el recibo de compatibilidad:
673 pruebas con dependencias conversacionales y 12 pruebas de ejemplos aprobadas;
la instalación mínima también pasó (577 pruebas, incluidas las omisiones por dependencias).

## Reproducción y evidencia

- Código medido UNIDAD: `ecf03da85e129cb94c46524a88b32cabc5569e81`.
- Extensión reconciliada NERVA: `1b9f84bdeb8a164d21bd2cee9f05f5150a90eef9`.
- [Corrida completa](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38445054650).
- [Resumen estructurado](summary.json) y [hashes/compatibilidad](compatibility.json).
- [C7 voraz](c7-greedy.json), [SOVENGI predeterminado voraz](sovengi-current-greedy.json),
  [candidato voraz](sovengi-candidate-greedy.json).
- [C7 muestreo](c7-sampling.json), [SOVENGI predeterminado muestreo](sovengi-current-sampling.json),
  [candidato muestreo](sovengi-candidate-sampling.json).
- [Consultas](cases.json), [selector Node](selector-kernel.json), [Python/Node](selector-ipc.json).

El workflow `nerva-selector-benchmark.yml` conserva parámetros y comandos. Las
versiones del entorno, CPU, hashes de código y artefactos están dentro de los informes.
Los tiempos son de CPU en GitHub Actions, no de Android, GPU ni del dispositivo del creador.
Tres/cinco rondas describen esta máquina y estas consultas; no prueban una aceleración universal.

**Corrección histórica:** los 4,368 s y 27,131 s de la comparación anterior eran
`medianSuiteSeconds` de diez consultas, no el tiempo de una sola respuesta. Sus
resultados originales permanecen en `../nerva-conversation-20261013/`. No se comparan
tiempos absolutos de máquinas o versiones distintas como si fueran el mismo ensayo.

La entrega conserva `nerva-fast` como opción explícita y Torch con caché como
predeterminado. Ninguna de las dos ramas se fusionó automáticamente.
