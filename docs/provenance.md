# Procedencia de la primera extracción

Unidad es un proyecto independiente. El especialista de atención, sus pruebas,
las capturas públicas y el primer artefacto proceden del repositorio del usuario
[Hormiga-X](https://github.com/guadalupejuan001-code/Hormiga-X), commit
[`2e7c002117ff4a44d2676bc143f33145fbb79973`](https://github.com/guadalupejuan001-code/Hormiga-X/tree/2e7c002117ff4a44d2676bc143f33145fbb79973).
La extracción se realizó el 2026-10-10 y no modificó ese repositorio.

El paquete usa imports locales `ia_propia.trend`; sus pruebas y ejemplos usan
archivos de este proyecto. No importa ni necesita una instalación de Hormiga-X,
su runtime, Android, Kotlin, herramientas, memoria o modelos de lenguaje.
La ruta del CLI y su texto visible se adaptaron. La implementación numérica, las
reglas de validación y el contrato JSON se conservaron. Las pruebas cambiaron
imports, raíz del proyecto, rutas de fixtures/evidencia y nombre del módulo CLI.

El artefacto entrenado y el informe histórico se copiaron **byte a byte**, sin
renombrar campos internos ni reescribir resultados. El formato mantiene
`schemaVersion=1`, `modelType=hormiga-trend-mlp` y `featureVersion=log-deltas-level-v1`.
Son identificadores del contrato heredado, no una dependencia de Hormiga-X.
Modelo preservado: `hormiga-trend-d3c54fb589a41b774b8e015e972d69403e14665c648dcaeae3ce03affa874da0`. Inició con pesos aleatorios y no usa
pesos preentrenados. El informe conserva la selección por validation, época
38, y el resultado test **peor que persistencia**
(RMSE 0.178261273
frente a 0.133980983).

## SHA-256 de los archivos

El hash de origen se calculó sobre el contenido exacto de `git show` en el commit
indicado y se contrastó contra el archivo de origen. El hash de destino se calculó
sobre los bytes actuales guardados en Unidad. El CLI ya incluye el nombre elegido
y la ayuda con el prefijo completo de comando; los pesos se conservan intactos. Los hashes distintos de CLI/pruebas
corresponden a las adaptaciones descritas arriba.

| Archivo original en Hormiga-X | Archivo en Unidad | SHA-256 original | SHA-256 actual |
| --- | --- | --- | --- |
| `training/hormiga_trend/__init__.py` | `src/ia_propia/trend/__init__.py` | `3d3b3df22651bf904b8dbe38ad75509355be6fcd86fe099abff5fc4d4aa6e7e8` | `3d3b3df22651bf904b8dbe38ad75509355be6fcd86fe099abff5fc4d4aa6e7e8` |
| `training/hormiga_trend/data.py` | `src/ia_propia/trend/data.py` | `609b3538d4276ec70171f79a3a9fc0ad3a6673f835d5f4f2aecef0686a699111` | `609b3538d4276ec70171f79a3a9fc0ad3a6673f835d5f4f2aecef0686a699111` |
| `training/hormiga_trend/model.py` | `src/ia_propia/trend/model.py` | `b8aa67252e22286ee19a1bea9bb8bc4a15df686c6217a214e7c817fa66e79261` | `b8aa67252e22286ee19a1bea9bb8bc4a15df686c6217a214e7c817fa66e79261` |
| `training/hormiga_trend/__main__.py` | `src/ia_propia/trend/__main__.py` | `d58808f64bf2ad9cfb562e353b30f58152ccba7a26698df3e3ed43d1f4227f30` | `95f63c9a4f025a69aa125e76835b8eab3fb1b6759ce230ef5911c70bf1b3d966` |
| `training/tests/test_training.py` | `tests/test_trend.py` | `6959e139bc9b93db3c060bd1ff7035fdce69e9d2741a015beae17447ad9fe59b` | `6ed36f08ec504344d23e614c47dc233544bcc2e245b1d521583b8e2086af6992` |
| `training/tests/fixtures/parity.json` | `tests/fixtures/parity.json` | `d848ac161ee22a8d80511f2aa74f89e279c2e2bce73e10872d28afa334a856a4` | `d848ac161ee22a8d80511f2aa74f89e279c2e2bce73e10872d28afa334a856a4` |
| `docs/evidence/real-observations.json` | `examples/attention-observations.json` | `2e74338724a145130a6d82a9bba64e2ed8ef4651da35fc63208d80636679586c` | `2e74338724a145130a6d82a9bba64e2ed8ef4651da35fc63208d80636679586c` |
| `docs/validation/own-trend-model-0.1.14.json` | `models/attention-v1.json` | `3703c3c89f8a2a791819772b973abeecb7951a4e28730273a5dbc1832a7baa2e` | `3703c3c89f8a2a791819772b973abeecb7951a4e28730273a5dbc1832a7baa2e` |
| `docs/validation/own-trend-training-0.1.14.json` | `docs/attention-training-v1.json` | `af6c076aafc7d522712065a348ac8da8026220721331b516a908cbf2885ccae4` | `af6c076aafc7d522712065a348ac8da8026220721331b516a908cbf2885ccae4` |

El hash del dataset registrado dentro del artefacto es
`bbcee978440c71aebbb5233c9d5d283bfeb90af13dcadad8d82d461158de3c9b`. Identifica la serialización canónica
ordenada de las observaciones seleccionadas; no es el hash de los bytes del
archivo JSON con espacios y saltos de línea.

## Evidencia pública original

`examples/attention-observations.json` contiene 30 capturas PUBLIC de visitas
Wikimedia para Moo Deng y Wild Rift en octubre de 2024. Cada captura conserva
fuente, URI, SHA-256 del contenido, `observedAt` y `fetchedAt`. La fecha medida es
el fin del intervalo diario; la recuperación ocurrió en 2026. El entrenamiento
es retrospectivo y no prueba que el sistema predijera en 2024.

Las respuestas Wikimedia originales permanecen disponibles en el repositorio de
origen, junto con su documentación de importación:

- [docs/evidence/moo-deng-2024-10-pageviews.json](https://github.com/guadalupejuan001-code/Hormiga-X/blob/2e7c002117ff4a44d2676bc143f33145fbb79973/docs/evidence/moo-deng-2024-10-pageviews.json): SHA-256 `18b5baa4f2afa1606637a859253f815aa319f014496e844bd585fc420ae2d960`.
- [docs/evidence/wild-rift-2024-10-pageviews.json](https://github.com/guadalupejuan001-code/Hormiga-X/blob/2e7c002117ff4a44d2676bc143f33145fbb79973/docs/evidence/wild-rift-2024-10-pageviews.json): SHA-256 `49a7e6e24cb62be59b1f603fad4accf37c8675709c7273ae79150fdad1c158d4`.

Estas dos respuestas no son dependencias de ejecución del nuevo paquete: las
capturas importadas y la procedencia requerida ya están incluidas. El test
numérico `tests/fixtures/parity.json` es un fixture fijo, no evidencia cognitiva ni
un modelo entrenado de producción. Las 20 pruebas extraídas validan aprendizaje
sobre una serie funcional, causalidad, privacidad, roundtrip y el recorrido real;
no convierten seis ventanas de test en aceptación de generalización.
