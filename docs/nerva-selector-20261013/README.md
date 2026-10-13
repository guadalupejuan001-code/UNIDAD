# Evaluación de la ruta opcional NERVA

Base UNIDAD: `2ad7b6efbe5f35759241649537243da1fb7bc3f5`.
Base NERVA PR #1: `b3002995590aa8b960749e6258acddbf67c3897f`.
Rama independiente en ambos: `work/nerva-selector-performance-20261013`.

`nerva-fast` es una opción explícita del anfitrión. Los valores predeterminados,
la API del intérprete, su CLI y sus siete archivos originales se conservan.
La extensión ejecuta exclusivamente el selector `.nv` identificado por hash,
reutiliza las primitivas NERVA f64 y mantiene sus cargos de operaciones,
límites de pasos, entradas, salida y proceso. No se entrena ni modifica un peso.

Comparación prevista sobre la misma máquina:

1. Selector dentro de Node: intérprete público frente a especialización,
   análisis del programa y lectura JSON, con cinco rondas y calentamiento.
2. Selector Python/Node: conversión de puntuaciones, JSON, IPC con datos ya
   codificados y selección completa. Incluye tiempos de arranque separados.
   Son mediciones independientes; sus medianas no son una descomposición aditiva.
3. Generación completa: diez consultas conocidas × tres repeticiones × cuatro
   rutas (Torch sin caché, Torch con caché, NERVA y NERVA especializada con caché).
   Tres artefactos: C7 histórico, SOVENGI predeterminado `reference-wiki1` de
   15,77 M parámetros y candidato de memoria reciente, sin promover este último.
4. Muestreo: tres consultas × tres repeticiones, temperatura 0,7, top-k 40,
   top-p 0,9 y penalización 1,1. Se exige igualdad entre las dos rutas NERVA.
   Torch utiliza otro RNG: no se exige igualdad de muestras entre esos motores.

Los informes conservan cada salida, sus IDs de tokens, terminación, tiempos,
hashes de código y de artefactos antes/después. El coste de respuesta incluye
arranque, IPC y cierre de Node; excluye la carga inicial del modelo compartido.
La selección y la generación se presentan por separado. No se evalúa aquí la
calidad conversacional ni la generalización de los modelos.

**Corrección de unidades históricas:** en
`../nerva-conversation-20261013/benchmark-c7.json`, 4,368 s (Torch) y 27,131 s
(NERVA) son `medianSuiteSeconds`: el conjunto de diez consultas, no una respuesta.
Las nuevas métricas incluyen tiempo del conjunto y mediana por respuesta.

Ejecución reproducible: `.github/workflows/nerva-selector-benchmark.yml`.
Los JSON medidos se incorporarán aquí al terminar la corrida. No se cambia el
backend predeterminado por los resultados de un prototipo aislado.
