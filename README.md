# Unidad

**Proyecto de IA propia, local y experimental, con modelos entrenados desde cero
y un cerebro compartido para distintas aplicaciones.**

Unidad reúne pesos propios, entrenamiento, evaluación, memoria persistente y
herramientas configuradas por la aplicación. La visión es desarrollar una IA
personal que converse en español, realice trabajo útil y pueda adaptarse mediante
aprendizaje verificable. La inferencia actual funciona localmente con Python,
NumPy y PyTorch, sin llamar a una API de otra IA.

**Estado del 13 de octubre de 2026:** GNOSIS y la integración NERVA están en
`main`. El código funciona y tiene pruebas, pero la conversación fiable sigue
pendiente. No hay evidencia de sustitución de asistentes comerciales,
autoaprendizaje o programación autónoma.

## Qué existe y funciona hoy

| Componente | Capacidad implementada | Alcance actual |
| --- | --- | --- |
| Cerebro compartido | `Brain` y `ConversationEngine` sirven los mismos modelos desde Python, CLI y HTTP. | Cada producto configura herramientas y memoria; el modelo no recibe acceso automático a cuentas. |
| Modelos propios | Pesos, configuración, tokenizador y procedencia de Transformers entrenados desde pesos aleatorios. | Son modelos experimentales; C7 se conserva como referencia. |
| GNOSIS | Una configuración reúne cerebro, NERVA, perfiles, recursos y etapas `foundation → conversation → adaptation`. | Las etapas requieren datos y ejecución explícitos; no deciden solas qué aprender. |
| NERVA 0.2 | Backend del modelo pequeño de caracteres, con generación greedy y temperatura cero. | Usa Node.js; C7 y los Transformers siguen ejecutándose mediante PyTorch. |
| Entrenamiento | Preentrenamiento de texto, ajuste conversacional, continuación desde pesos propios y reanudación con optimizador. | Datos separados por documento/familia, hashes y checkpoints; la configuración GNOSIS entregada todavía no fija corpus. |
| Memoria local | Registro de encargos, intentos y correcciones, con búsqueda y referencias al texto original. | Guardar memoria no cambia pesos ni incorpora recuerdos automáticamente al diálogo. |
| Herramientas y flujos | Ejecuta operaciones y herramientas registradas mediante pasos explícitos. | Los pasos los aporta la aplicación; todavía no existe un planificador aprendido. |
| Especialistas pequeños | Generación por caracteres y predicción de una métrica de atención. | Son demostraciones con errores y límites documentados, no capacidades generales aceptadas. |

Las rutas privadas se excluyen de Git. La selección de modelos, herramientas,
backends y recursos corresponde al anfitrión; las peticiones no pueden cambiarlos.
La API permite construir otros clientes, incluido Android, pero esta integración
no conecta automáticamente una aplicación Android ni acredita una prueba física.

## Los modelos: entrenamiento y escalado son cosas distintas

| Modelo o perfil | Parámetros | Qué se comprobó |
| --- | ---: | --- |
| [C7](models/conversation-reference-c7/README.md) | 6.983.937 | Referencia propia entrenada, con pesos publicados, vocabulario BPE de 8.192 tokens y contexto de 512. Su conversación no pasa la aceptación. |
| `50m` | 48.852.992 | Un modelo real inicializado aleatoriamente calculó gradientes de 64 tokens, se exportó y se cargó con resultados idénticos. Tuvo **cero actualizaciones de pesos**. |
| `100m` | 99.446.400 | Perfil validado y conteo exacto. No se ejecutó ni entrenó aquí. |
| `1b` | 991.586.304 | Perfil validado y estructura meta, sin asignar sus pesos en CPU. No se ejecutó ni entrenó aquí. |

**El modelo de 49 millones no es C7 ampliado ni un modelo que haya aprendido
conversación.** Fue una prueba local de cálculo y manejo de artefactos. En `main`
están el [runner reproducible](examples/verify_gnosis_scaling.py) y su
[evidencia](docs/gnosis-scaling-20261013.json); no se publicó allí un checkpoint
conversacional entrenado de ese tamaño.

El código admite hasta 1.200 millones de parámetros mediante un
`parameter_budget` explícito. Las configuraciones anteriores conservan el límite
de 20 millones. Un perfil no proporciona el hardware, los datos ni las horas de
entrenamiento que necesita. El entrenamiento actual es CPU. El cargador opcional
`data_loading="mmap"` prepara corpus y ventanas en disco para muestreo uniforme;
GPU sigue pendiente. Los [controles y mediciones del cargador](docs/tokenized-data-20261013.md)
no acreditan una mejora de las respuestas.

## Qué dicen las evaluaciones

La prueba reservada más reciente utilizó **40 turnos en diez diálogos**, historia
realmente generada, 160 tokens por respuesta, temperatura cero y semilla 42.

| Modelo | Respuestas completas | Diálogos completos | Fallos críticos |
| --- | ---: | ---: | ---: |
| C7 | 0/40 | 0/10 | 4 |
| Ajuste parcial de copia, 128 pasos | 0/40 | 0/10 | 8 |

Las revisiones primaria y cruzada coinciden en estos recuentos. Son revisiones
de IA del proyecto, **no evaluaciones humanas independientes**; se conservan sus
desacuerdos sobre respuestas parciales. No hubo fallos técnicos en esos 80 turnos.
El candidato parcial fue rechazado y C7 conservó sus pesos e identidad.

La aceptación exige **36/40 respuestas completas, 8/10 diálogos completamente
correctos y cero fallos críticos o técnicos** en una prueba nueva. No se alcanzó.
El resultado histórico de C7 de 7/40 corresponde a otra batería: no debe mezclarse
con esta prueba ni presentarse como mejora entre ambas.

La [comparación completa](docs/brain-nerva-20261013/reserved-comparison.json)
conserva respuestas, rúbrica, condiciones e identidades. El
[estado actual](docs/ESTADO_ACTUAL.md) reúne las fuentes y los experimentos anteriores.
Todavía no hay una comparación que demuestre calidad competitiva frente a una IA comercial.

## Qué se optimizó de verdad

Se incorporaron microbatches, escritura de artefactos por bloques y carga sin
inicializar otra copia completa del modelo. La reanudación se comprobó en FP32 y
BF16, también después de un fallo en mitad de un lote. La admisión fija corpus y
arquitectura antes de cargar pesos y rechaza rutas de etapas que se solapen.

En la medición CPU del modelo aleatorio de 48,85 millones de parámetros:

| Operación | Pico RSS nuevo | Pico RSS de referencia |
| --- | ---: | ---: |
| Exportación | 651 MB | 1.237 MB |
| Carga | 710 MB | 876 MB |

Son observaciones específicas, no un ahorro universal; el RSS residente final de
la carga no mejoró. En una prueba pequeña separada, los microbatches redujeron un
**87,46 %** las reservas de activaciones retenidas por autograd, conservando la
pérdida con diferencias mínimas de gradientes. Esa cifra no representa la RAM total.

NERVA reprodujo **448 caracteres** de NumPy en siete generaciones; el arranque de
Node añadió tiempo. No se demostró aceleración con ese backend.

La verificación local del cargador en disco aprobó **462 pruebas y 12 pruebas de
ejemplos**; el [recibo](docs/tokenized-data-20261013/software-verification.json)
identifica código y comprobaciones. La integración anterior de GNOSIS aprobó
441 pruebas y también [GitHub Actions](https://github.com/guadalupejuan001-code/UNIDAD/actions/runs/38332550358).
Estas comprobaciones acreditan funcionamiento técnico, no comprensión ni calidad conversacional.
Las [mediciones y límites](docs/gnosis.md) y la
[verificación de software](docs/gnosis-evidence/software-verification.json)
conservan el detalle.

## Probar el proyecto

Python 3.11 o posterior. Desde un clon del repositorio, en Linux/CPU:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[conversation]'
unidad --help
```

Para usar la configuración GNOSIS con su backend NERVA, instala también
**Node.js 18 o posterior**. Después:

```sh
unidad gnosis plan --project configs/gnosis.json
unidad gnosis capabilities --project configs/gnosis.json --body pc
unidad gnosis chat --project configs/gnosis.json \
  --max-new-tokens 160 --temperature 0 --seed 42
```

El chat carga C7. `plan` calcula recursos sin crear pesos ni iniciar entrenamiento;
la configuración entregada informa que los corpus están sin configurar. Clonar
pesos y checkpoints no inicia un proceso que siga aprendiendo.

Desde Python, dos productos pueden compartir el mismo cerebro:

```python
from ia_propia.gnosis import load_project

project = load_project("configs/gnosis.json")
pc = project.connect("pc")
android = project.connect("android")
# Mismo cerebro y modelos; memoria separada por producto.
```

La [guía GNOSIS](docs/gnosis.md) explica las etapas, los formatos de datos,
presupuestos, exportación y evaluación. La [guía del cerebro](docs/brain.md)
describe el protocolo HTTP y las herramientas por producto. Los datos personales
y modelos entrenados con ellos deben conservarse en `private/` o `local-models/`.
Las órdenes y los mensajes de este chat no se usan automáticamente como corpus.

## Potencial y prioridades

El potencial de Unidad está en poder controlar sus datos, pesos y evolución:
entrenar una base propia, especializarla en tareas delimitadas y utilizarla desde
varias aplicaciones mediante el mismo núcleo. GNOSIS permite organizar nuevas
etapas sin rehacer cada interfaz y conservar la procedencia de lo aprendido.
La memoria y las herramientas ofrecen una base para construir un asistente
personal útil cuando el modelo pueda comprender y realizar esas tareas de forma fiable.

Ese potencial depende de resolver la calidad del aprendizaje. La prioridad es:

1. Ampliar y diversificar la base lingüística de español, conservando procedencia,
   derechos de uso y separación entre entrenamiento, validación y evaluación.
2. Conseguir conversación que mantenga hechos, siga correcciones y responda a la
   petición actual; medirla con diálogos nuevos e historias realmente generadas.
3. Escalar datos, contexto y modelos con recursos adecuados, utilizando el
   cargador en disco e incorporando GPU.
4. Comprobar adaptación y retención entre etapas antes de promover nuevos pesos.
5. Evaluar después herramientas, tareas completas y propuestas de modificación
   de código, cada una con criterios propios.

Acercarse a asistentes comerciales es un objetivo de investigación, no un
resultado garantizado por usar NERVA, aumentar parámetros o dedicar más horas.
La adaptación automática y la autoprogramación forman parte de la visión futura;
necesitan demostraciones específicas y todavía no están implementadas como
capacidades autónomas del modelo.

## Mapa del repositorio y sus propuestas

Este repositorio también conserva desarrollos fuera de `main`. Al comprobar
GitHub el 13 de octubre de 2026 UTC había **ocho ramas y seis PR**: PR5 y PR6
estaban fusionados; PR1–4 seguían abiertos y ya no figuraban como borradores.

| Ubicación | Qué aporta | Situación comprobada |
| --- | --- | --- |
| `main` | Cerebro, modelos propios, entrenamiento, evaluación, GNOSIS, NERVA y datos en disco. | GNOSIS/NERVA se integraron por [PR5](https://github.com/guadalupejuan001-code/UNIDAD/pull/5); el cargador mmap, por [PR6](https://github.com/guadalupejuan001-code/UNIDAD/pull/6). |
| [PR1](https://github.com/guadalupejuan001-code/UNIDAD/pull/1) | Diagnóstico de conversaciones, correcciones del evaluador y evidencia experimental. | Abierto; parte de las correcciones ya llegó a main por otra vía. |
| [PR2](https://github.com/guadalupejuan001-code/UNIDAD/pull/2) | Proyección de vocabulario sólo en la posición final durante generación. | Optimización propuesta; sus experimentos de conversación no justificaron promover pesos. |
| [PR3](https://github.com/guadalupejuan001-code/UNIDAD/pull/3) | Reparación de redacción y contraste relacional desde C7. | Candidatos rechazados; conserva respuestas y evaluaciones. |
| [PR4](https://github.com/guadalupejuan001-code/UNIDAD/pull/4) | Exportador de un paquete de modelo para Android y comprobaciones con ONNX Runtime. | Propuesta abierta; la paridad en CPU no acredita funcionamiento en un teléfono físico. |
| [Rama de artefactos](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012) | Paquete dividido en seis partes: C7, candidatos relacionales y checkpoints completos. | Conserva experimentos y permite reconstruirlos; no convierte candidatos rechazados en modelos aceptados. |

Los recibos y textos antiguos conservan el estado de su fecha. Para saber si una
propuesta está integrada o sigue abierta, consulta también su PR.

NERVA incluye además un lenguaje e intérprete con operaciones de tensores,
diferenciación automática, actualización de parámetros y checkpoints. Es una
pieza de experimentación que merece describirse por sí misma. Su adaptador actual
dentro de Unidad sólo realiza inferencia del pequeño modelo de caracteres; no
entrena C7 ni ejecuta su Transformer.

## Documentación y antecedentes

- [GNOSIS: configuración, comandos y límites](docs/gnosis.md)
- [NERVA dentro del cerebro](docs/nerva.md)
- [Entrenamiento conversacional](docs/conversation.md) y [base lingüística](docs/language-pretraining.md)
- [Memoria y encargos](docs/workspace.md) y [núcleo para productos](docs/brain.md)
- [Estado y fuentes](docs/ESTADO_ACTUAL.md) y [estado estructurado](docs/unidad-current-status.json)
- [Órdenes del creador](ORDENES_USUARIO.md) e [instrucciones de desarrollo](AGENTS.md)
- [Antecedentes y comandos del README anterior](docs/historial-readme-20261013.md)

Los pesos y recibos de experimentos descartados se conservan como evidencia.
Publicarlos no los convierte en modelos aceptados. Los presupuestos anteriores
de entrenamiento están cerrados; esta documentación no inicia ni renueva corridas.
