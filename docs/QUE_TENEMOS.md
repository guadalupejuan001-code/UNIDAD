# Qué tenemos en Unidad

La entrada unificada y la IA se llaman **SOVENGI**; UNIDAD conserva el nombre
del repositorio y del paquete por compatibilidad. La [guía de SOVENGI](SOVENGI.md)
explica su configuración actual, memoria y API.

Tenemos un proyecto de IA propia que puede entrenarse y ejecutarse localmente,
con un núcleo para conectarlo a distintas aplicaciones. Hay código funcional,
pesos entrenados, datos con procedencia y pruebas conservadas. La misión de
conseguir un asistente personal que converse y trabaje con fiabilidad sigue pendiente.

El [entrenamiento de memoria más reciente](sovengi-memory-20261013/README.md)
completó 27 actualizaciones y mejoró un control acotado de 1/25 a 22/25 respuestas
exactas. Los pesos nuevos pueden elegirse en SOVENGI con
`--config configs/sovengi-memory-candidate.json`. El ejercicio de notas guardadas
mejoró; la conversación abierta y el reconocimiento de información desconocida
todavía requieren trabajo. El informe conserva las respuestas anteriores y nuevas.

Como antecedente, el [PR11](https://github.com/guadalupejuan001-code/UNIDAD/pull/11) se fusionó el
13 de octubre de 2026 a las 12:10:20 UTC en `a948c0fd…`, con cuatro checks
aprobados. Eso verifica la entrega de software de ese momento; la calidad del
modelo se comprueba aparte con sus respuestas y trabajos reales.

Los modelos de conversación son Transformers propios: su origen son pesos
aleatorios, y las continuaciones parten de checkpoints de esa misma familia.
C7 se conserva como referencia propia; también hay una base de **15.179.520
parámetros** y variantes de **15.769.344** con contexto de **2.048 tokens**.
C7 es una etiqueta del candidato anterior; la IA se llama SOVENGI y el repositorio
se llama UNIDAD. La base de quince millones comenzó con nuevos pesos aleatorios
propios; sus continuaciones reutilizan esa línea y no redimensionan C7.
No se usan los pesos de otro asistente para reemplazarlos. Un parámetro es un
número que el entrenamiento ajusta; tener más no garantiza mejores respuestas.

El tokenizador BPE divide el texto en fragmentos: un token puede ser parte de una
palabra. El contexto es la cantidad de esos fragmentos que el modelo puede leer
a la vez. Primero se entrena a predecir texto de documentos —aprendizaje causal—
y después a responder al usuario —ajuste supervisado o SFT—. Los archivos guardan
qué datos y configuración produjo cada versión. [La guía de conversación](conversation.md)
y [la de preentrenamiento](language-pretraining.md) explican esos recorridos.

La atención y las capas neuronales ya permiten aprender patrones. El
[ajuste específico de relaciones](learning-repair-20261013/pattern-relations-v2/README.md)
ejecutó 512 actualizaciones y pasó de cero a once completas en doce ejemplos
entrenados. Sólo logró una completa y una parcial de doce prefijos nuevos,
según revisión de IA, y ninguno de los dos diálogos fue correcto. Hay aprendizaje
del control, pero siguen fallando atribución de hechos y generalización.
Entregar estos pesos experimentales no significa que UNIDAD converse bien.

| Pieza | Para qué sirve |
| --- | --- |
| [Brain](brain.md) | Núcleo compartido: carga los modelos y ofrece operaciones de conversación, memoria y herramientas a cada aplicación. |
| CLI, Python y API HTTP | Tres formas de usar el mismo núcleo. El servidor recibe peticiones autenticadas; cada producto conserva sus recursos y permisos. |
| `ConversationEngine` | Motor común que genera respuestas con los pesos seleccionados. Lo utilizan consola, Brain y evaluación. |
| [GNOSIS](gnosis.md) | Configura modelos, recursos y etapas de base, conversación y adaptación. Es coordinación del proyecto; consultar su plan no entrena. |
| [NERVA](nerva.md) | Ejecuta el modelo de caracteres y puede elegir tokens del Transformer. Los pesos calculan sus probabilidades en PyTorch; NERVA aplica la política de selección. |

Esta separación permite conectar un editor, un panel u otro producto sin copiar
el modelo para cada uno. Los clientes existentes conservan su contrato de
operaciones; añadir autonomía ofrece operaciones nuevas sin obligarlos a usarlas.
La aplicación decide qué modelo, herramientas y memoria conecta. Una petición
del usuario o una respuesta del modelo no concede permisos nuevos.

La [integración conversacional de NERVA](nerva-conversation-20261013/README.md)
también añade una caché: reutiliza cálculos del contexto mientras la respuesta
crece. En diez consultas C7, tres repeticiones y temperatura cero, la ruta Torch
con caché fue 2,91 veces más rápida y conservó los mismos tokens. NERVA funcionó,
pero añadió tiempo; queda como opción del anfitrión y la ruta más rápida sigue
siendo la predeterminada. Estas opciones no cambian lo aprendido por el modelo.

La [memoria local](workspace.md) registra encargos, intentos y correcciones y
permite buscarlos. SOVENGI añade al contexto notas relevantes de la misma
memoria, si caben, y devuelve sus fuentes. **Guardar una nota no modifica los pesos** ni garantiza que
el modelo la recuerde en otro diálogo. Exportar ejemplos aceptados para un
entrenamiento posterior es una operación distinta. Los datos personales y las
credenciales permanecen en rutas privadas excluidas de la publicación.

Hay [herramientas y un controlador operativo](learning-repair-20261013/operational-autonomy.md):
el modelo propone una acción JSON; el controlador valida argumentos y permisos,
ejecuta una herramienta autorizada y verifica el resultado. Conserva progreso
en SQLite, límites, cancelación y recuperación. Las acciones sensibles requieren
aprobación vinculada a una acción concreta, un nonce y credencial de operador.
Cancelar impide pasos nuevos; no promete deshacer efectos externos anteriores.

El controlador sí completó el análisis de archivos y un informe con propuestas
de prueba explícitas, llamadas *fixtures*. Eso comprueba el software y el IO.
Los intentos del modelo propio se conservan separados: uno inspeccionó realmente
un archivo autorizado, pero después produjo JSON inválido y no terminó su informe.
En [dos tareas DEV posteriores](learning-repair-20261013/action-copy-v2-development-tasks-v1/README.md),
tratamiento y control completaron **0 de 2** cada uno. La planificación fiable
todavía no está demostrada; una secuencia programada no la sustituye.

Los [datos de español](../examples/learning-repair-20261013/DATASOURCES.md) incluyen
fuentes públicas, textos de práctica y sus particiones. Se conservan revisiones,
atribución, hashes y [licencias propias de cada fuente](../examples/learning-repair-20261013/LICENSE-DATA.md).
El [material latinoamericano](../examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/README.md)
añade vocabulario, usos y diferencias por país. Es una muestra incompleta;
reunir títulos de entradas no equivale a reunir todos sus significados ni a que
el modelo los haya aprendido. Los ejemplos sintéticos se identifican como tales.

El [supervisor local](learning-repair-20261013/loop-runtime.md) encadena etapas con
datos admitidos, checkpoints y límites finitos; genera respuestas para revisión.
Su collector obtiene lotes nuevos de Wikipedia con procedencia y particiones.
Ya ejecutó un lote nuevo y pasó a retención. El entrenamiento observado usa
cuatro hilos de CPU, sin GPU. Depende de que el proceso, el entorno, los datos y los
recursos sigan disponibles. No promueve modelos automáticamente ni convierte
cada charla en entrenamiento; clonar el repositorio tampoco inicia el bucle.

Podemos instalar el paquete, abrir la consola del modelo propio, integrar Brain
en Python o servir su API. Desde la raíz, después de instalar las
[dependencias CPU](conversation.md#instalar-para-cpu), se abre SOVENGI con el
modelo seleccionado en `configs/gnosis.json`:

```sh
sovengi chat --max-new-tokens 96 --temperature 0 --seed 42
```

También se pueden consultar planes GNOSIS, registrar memoria, ejecutar flujos
explícitos y reproducir las demostraciones siguiendo sus guías. Hay además un
[pequeño especialista de métricas](attention.md), con alcance separado; no es
una capacidad general para interpretar al usuario.

`models/` conserva artefactos de inferencia; `checkpoints/`, estados para recuperar
entrenamientos, incluidos optimizador y estado aleatorio; `examples/`, recetas y datos públicos;
`docs/`, explicaciones, resultados y recibos. Los hashes permiten comprobar que
una evaluación corresponde a unos pesos concretos; los experimentos fallidos
permanecen como historia y no se cambian por respuestas redactadas después.

La [comparación anterior a Tatoeba](learning-repair-20261013/comparison/development-wide-latam-retention1024-v1/README.md)
obtuvo 1/40 respuesta completa y ningún diálogo completo. La
[evaluación posterior a Tatoeba](learning-repair-20261013/comparison/development-tatoeba-retention1024-v1/README.md)
retrocedió a **0/40 completas y 0/10 diálogos**, con nueve/diez críticas según
el revisor y cinco técnicas. Son pruebas DEV examinadas y juicios de IA del
proyecto, sin panel humano independiente. Las generaciones de etapas posteriores
se revisan aparte; guardar más pesos no garantiza mejora. La aceptación exige
36/40, 8/10 y cero fallos críticos o técnicos en una prueba final nueva tras fijar
candidato y generación. Tenemos la infraestructura para seguir entrenando y
comprobar el resultado; falta demostrar conversación y trabajo fiables.

