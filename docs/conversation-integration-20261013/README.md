# Integración y aprendizaje conversacional de Unidad

El código compatible de las ramas se reúne en `main`, con pesos propios y
evidencia preservada. Se completaron seis ajustes y **2.560 actualizaciones de
Adam**. La mejora conversacional buscada **no se consiguió**: C7 y el candidato
congelado obtuvieron cero respuestas completas en la prueba final nueva. C7
conserva sus cuatro archivos e identidad y los nuevos modelos siguen
experimentales. No quedan entregas en borrador ni entrenamientos activos de esta
etapa. Este informe corresponde al encargo ampliado del 13 de octubre de 2026.

## Repositorio y código reconciliado

Se inspeccionaron las ocho ramas remotas originales, PR abiertos y fusionados,
modelos, checkpoints, scripts y evidencia. PR5/6 ya estaban integrados; PR1–4
estaban abiertos y se retiró su condición de borrador. Durante el trabajo llegó
`7990c4562852fda6ca366c058aacb1f781dda551`, una actualización ajena del README;
se incorporó antes de actualizar la presentación final.

La [matriz](branch-reconciliation.json) fija las cabezas y hashes de 324
documentos históricos importados, sin cambiar sus bytes. El evaluador de PR1/3
ya era idéntico al de `main`; se evitó duplicarlo. PR2 aporta la proyección de
vocabulario sólo en la posición final al generar. PR4 aporta el exportador ONNX,
corregido para comprobar longitud, impedir sobrescritura y rechazar mutaciones
de la fuente antes de publicar. GNOSIS, NERVA, mmap, carga sin doble
inicialización y el resto de mejoras posteriores se conservan.

El [recibo de ejecución](runtime/README.md) distingue optimización del cálculo y
paridad numérica de calidad de las respuestas. El parche de copia v2 permanece
como evidencia, sin aplicar. Los candidatos anteriores siguen rechazados. La
rama `artifacts/unidad-contraste-relacional-20261012` conserva su paquete y
checkpoints históricos; no se sobrescriben las ramas para borrar el recorrido.
La integración de ancestría conserva el árbol reconciliado, evitando restaurar
versiones antiguas del núcleo al registrar los PR ya revisados.

## Comprobación del aprendizaje

[learning-mechanism-checks.json](learning-mechanism-checks.json) vincula controles
inspeccionables y pruebas de desplazamiento causal, roles, EOS, padding, máscaras,
última respuesta, gradientes, precisión, Adam y reanudación. Las cuatro frases
de español/nombres nuevos conservan round-trip exacto en el BPE propio de 8.192
tokens; conservar el texto no prueba entenderlo. Los pesos y el tokenizador
corresponden por configuración y hashes.

El control pequeño compartido ya estaba ejecutado en las ramas inspeccionadas:
los brazos comparables desde C7 y desde pesos aleatorios memorizaron **16/16
respuestas TRAIN a 64 pasos**. Una prueba de redacción posterior pasó de 7/8
exactas a 0/8 con paráfrasis; la revisión semántica fue 7/8 frente a 1/8. Se
conservan [salidas y planes originales](../work-experiment-20261012/), con fecha
y código originales. No se repitió un control que ya demuestra memorización ni
se atribuyó su ejecución al código nuevo. Esos resultados no son aceptación.

El fallo histórico `stale` frente a `planned` no se reprodujo en esta ejecución.
La prueba intacta `test_detached_planned_stop_and_invalid_worker_status` pasó:
observó proceso vivo/planificado, parada cooperativa con checkpoint y estado
`failed` para un corpus inválido. `stale` sigue identificando un trabajador
registrado que ya no vive; no se cambia a `planned` para encubrir su ausencia.
La [verificación final](software/verification.json) conserva los logs de
**494 pruebas y 12 ejemplos aprobados**. La restauración de los tres paquetes
seleccionados también [validó pesos, Adam y RNG](software/checkpoint-restoration.json)
en directorios nuevos, sin realizar entrenamiento adicional.

## Hipótesis y comparaciones ejecutadas

El [diagnóstico](training/diagnosis.md) encontró concentración de C7 en escenas
sintéticas: 74% del muestreo registrado, frente a 1% humano (34 ventanas).
V7 conserva mucha más exposición humana, pero tampoco conversa de forma fiable.
No se encontró un defecto de shift/roles que explique por sí solo los fallos.
La insuficiencia de base lingüística/composición es una hipótesis sustentada,
no una causa completa aislada ni una cifra demostrada de tokens necesarios.

1. **Exposición humana:** ajustes iguales de 256 pasos desde C7 y V7 con español
   público breve. La NLL mejora; la generación libre no recupera conversación.
   El corpus v3 excluye familias con autoidentificación ajena; preserva v1 y sus
   particiones. V3 aporta anclas a la etapa posterior.
2. **Historia generada:** dos brazos de 512 pasos, mismo C7, usuarios, objetivos,
   orden, presupuesto y validación ideal. Sólo cambia el historial TRAIN por
   respuestas reales anteriores de C7; la respuesta final correcta conserva la
   supervisión. Se registraron 576 generaciones, 368 prefijos cambiados. No hubo
   ganancia neta de completas; tampoco se demostró que ese historial sea la causa
   principal del fallo.
3. **Variedad de varios turnos:** dos brazos de 512 pasos desde C7 y V7, con
   120 familias ficticias escritas por tres agentes del proyecto sin exposición
   a la batería de evaluación, más 150 anclas humanas. Se supervisa sólo la última
   respuesta y EOS. Ninguno añadió respuestas completas; C7 mostró un bucle.

Cada etapa fijó corpus/arquitectura/código, origen propio, Adam nuevo para la
etapa, todos los parámetros y límite finito de pasos/tiempo. Las reanudaciones
permitidas mantienen Adam, muestreo y RNG; empezar desde pesos es otra operación.
Los [recibos originales](training/README.md) muestran las configuraciones y
paradas cooperativas. Se detuvo cada intervención sin mejora en generación;
menor NLL no activó una ampliación.

[training-summary.json](training-summary.json) registra 704,874 segundos nativos
sumados de las seis corridas (unos 11 min 45 s), incluida preparación/validación
del entrenador. La autoría, preparación externa, evaluación, revisión y entrega
consumen tiempo adicional. Se usaron cuatro hilos de CPU y BF16 de cálculo, con
pesos propios FP32, batch 8 y mmap. El entorno dispone de cinco CPU, 17 GiB RAM
y disco de 32 GB; no hay GPU utilizada ni gasto nuevo. No se midió el pico RSS
de cada corrida: no se atribuyen las mediciones anteriores del cargador al
entrenamiento completo. Los archivos publicados no exceden 100 MiB.

## Datos y reproducibilidad

[data/README.md](data/README.md) reúne procedencia, licencias, hashes y límites.
OASST1/2 se fijaron por revisión, con Apache-2.0 y separación por familia. Su
contenido público no está certificado factual y contiene respuestas inadecuadas.
Los datos sintéticos identifican autoría de IA del proyecto; no son diálogos del
creador ni conversaciones humanas independientes. Las salidas reales de C7
sólo sustituyen historia, nunca objetivos positivos erróneos.

TRAIN/VALIDATION se publican separados de TEST. Los preparadores conservan folds
anteriores, eliminan duplicados entre familias, registran longitudes y publican
atómicamente. No se ajustó un nuevo tokenizador ni se tokenizó TEST durante los
entrenamientos. Los comandos de preparación están en los README de cada corpus.

[Modelos](../../models/conversation-integration-20261013/README.md) publica los
seis snapshots seleccionados, sin modificar manifiesto/configuración/tokenizador
ni pesos. [Checkpoints](../../checkpoints/conversation-integration-20261013/README.md)
preserva BEST y ENDPOINT completos para historia generada y ambos ajustes
naturales. Las guías distinguen la API nativa reubicable del runner del piloto,
que fija rutas históricas. La restauración se verificó sin entrenar; no se afirma
continuidad numérica comprobada en otra máquina.

```sh
python -m ia_propia.conversation converse \
  --model models/conversation-integration-20261013/history-generated \
  --max-new-tokens 160 --temperature 0 --seed 42
```

El [recibo de consola](software/published-model-cli.json) carga exactamente el
modelo congelado, `unidad-conversation-be6414b4187c33e27aef54e67352aaa78393afb303f7b8daaf8c0eb7c2a558a2`.
La consola funcional no acredita comprensión.

## Evaluación final y bloqueo observado

[final-candidate-freeze.json](final-candidate-freeze.json) se escribió antes de
abrir la reserva; fija el snapshot384, cuatro hashes y generación. Desarrollo
mostró una sola completa y ninguna ganancia neta respecto a C7. Fue elegido para
una comprobación prospectiva por tener menos críticas/técnicos; omitir hechos o
producir incoherencias no constituye mejora factual.

La reserva nueva usa diez diálogos de cuatro turnos, 160 tokens, temperatura cero,
semilla42, CPU de un hilo e historial realmente generado. La revisión primaria
de IA del proyecto, no humana independiente, obtuvo:

| Modelo | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C7 | 0/40 | 4 | 36 | 0/10 | 2 | 0 |
| Historia generada384 | 0/40 | 1 | 39 | 0/10 | 2 | 0 |

Ambos fallan el umbral de 36/40, 8/10 y cero críticas/técnicas. En esos 80 turnos
no hubo vacías, recortes ni límites de tokens alcanzados. El candidato sigue
confundiendo hechos/personas, no mantiene correcciones y promete memoria sin
base; C7 también pierde referencias y hechos corregidos. La interpretación y
los desacuerdos de las revisiones se preservan en [evaluation/](evaluation/).
No se modificaron pesos, datos, selección o rúbrica usando esta reserva.
La cruzada ciega de otro agente de IA del proyecto contó candidato0/0/40 y
C7 0/2/38, con una crítica en cada modelo y cero técnicas. Coincide en cero
completas, cero diálogos y rechazo; las discrepancias sobre parciales y críticas
quedan por turno en la comparación, sin reemplazar las calificaciones primarias.

Las seis correcciones explícitas no recibieron ninguna respuesta completa en
ninguno de los modelos. La rúbrica consideró comprensibles en español 12/40
respuestas del candidato y 8/40 de C7, sin acreditar fidelidad al diálogo. El
conteo descriptivo de 4-gramas repetidos señaló tres turnos por modelo; dos
respuestas del candidato tuvieron secuencias de un mismo token de longitud al
menos tres. El bucle completo de generación duró 8,415 s frente a 9,075 s de C7
(promedios del bucle 0,210 y 0,227 s/turno, excluida la carga). No se midieron
latencias individuales ni pico RAM/RSS; estos promedios no son una medición de
latencia de una aplicación. [Las métricas](evaluation/reserved-descriptive-metrics.json)
conservan definiciones, tokens y resultados por habilidad.

**Bloqueo concreto:** el entrenamiento puede memorizar y reducir su pérdida,
pero la recuperación de hechos y la transferencia a peticiones/formulaciones
nuevas siguen fallando. Los ajustes evaluados no resuelven esa generalización.
No se demostró que ampliar capacidad o tiempo lo solucione. La siguiente etapa
requiere probar una base lingüística más sólida con controles de transferencia
antes de preferencias o capacidades autónomas; no hay un proceso ejecutándola.
Los [métodos y fuentes originales](methods-original-sources.md) distinguen
hallazgos publicados de las decisiones locales y evitan atribuir recetas
completas de productos actuales. No se implementó PPO, GRPO ni DPO sin datos
comparativos y evidencia que justificaran su coste.
