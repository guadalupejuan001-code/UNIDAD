# Descarga recuperada y entrenamiento en marcha

Los mensajes del usuario no interrumpieron las descargas. El servicio oficial
`datasets-server.huggingface.co/rows` respondió HTTP 502 durante la lectura de
Wikipedia en español. Los intentos fallidos conservaron la caché y el cursor;
no se perdieron pesos ni se reinició un entrenamiento.

El 13 de octubre de 2026 se completó el lote pendiente `wiki-cycle-0003`:

| Dato comprobado | Resultado |
| --- | ---: |
| Páginas lógicas, cada una con cobertura completa | 58, órdenes 728–785 |
| Páginas de caché reutilizadas | 28 |
| Solicitudes nuevas en la ejecución exitosa | 30, todas HTTP 200 |
| Bytes nuevos leídos de la fuente | 10.977.749 |
| Tiempo del recolector / proceso completo | 70,113 / 70,844 segundos |
| Registros nuevos / artículos únicos | 2.553 / 2.545 |
| Registros conocidos de repaso | 2.299 |
| Particiones del corpus resultante | TRAIN 4.325, VALIDATION 527, TEST 0 |
| Corpus resultante | 18.399.986 bytes |
| Cursor después de la publicación | 786 |

La revisión de la fuente es exactamente
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuración `20231101.es`.
Es un snapshot de noviembre de 2023, no noticias de 2026 ni un catálogo completo
del español. Se seleccionó prosa con las reglas ya existentes, conservando
familias, atribución, licencias y límites. Los 2.299 registros conocidos cuentan
como repaso, no como información nueva. No se decodificó texto de la partición
reservada de la fuente ni se abrió la evaluación final conversacional.

El [corpus distribuible y su manifiesto](../../../examples/learning-repair-20261013/wiki-cycle-0003-recovered/README.md)
son copias idénticas de los archivos utilizados por el trabajador. SHA256 del
corpus: `4dcd3a9e1f8573ff5030f37d7a0cd22768bd64d3dbdeb706594a8d7dbedb8483`.
Los cachés HTTP completos, que también transportan texto reservado sin
decodificar, permanecen privados.

## Qué se corrigió

El [transporte de recuperación](../../../examples/collect_learning_wiki_recovery.py)
permite hasta cuatro intentos por página y 320 solicitudes en una ejecución
finita. Reutiliza el bloqueo, caché y transacción del recolector V2. Comprueba
URL, HTTP real, revisión y todos los índices de fila antes de guardar una página.
Registra también errores, hashes y bytes; un fallo no se convierte en una
respuesta exitosa. El cursor avanza después de verificar y publicar el lote.

La ejecución exitosa no necesitó reintentos: sus treinta solicitudes fueron
HTTP 200 al primer intento. Las pruebas de errores/reintentos son fixtures
declaradas, separadas de estas descargas reales. Pasaron cinco pruebas nuevas y
siete del recolector original, con logs conservados en esta carpeta.

Se mantuvieron los límites de 900 segundos de colección, 512 MiB de lecturas
de red, 80 páginas, 24 MiB de corpus y 2 GiB de almacenamiento del colector.
El proceso usó un hilo CPU, prioridad reducida y un límite externo de 512 MiB
de espacio de direcciones. No se contrataron servicios ni se amplió un
presupuesto de entrenamiento cerrado.

## Incorporación real al modelo

El supervisor existente 22557, nacimiento `2592201`, admitió `wiki-cycle-0003`
y lanzó un único trabajador nativo. En el corte de las **15:18:20 UTC**, el
trabajador **33674**, nacimiento `4777437`, había ejecutado **251 actualizaciones**
de aprendizaje causal. Su corpus tiene el hash anterior y su padre es el
checkpoint propio `80f5c75026937e249b42a5aabde2a702034739d71bfeefe14b1dda7d786545ed`,
procedente de `wiki-cycle-0002:sft`. No se adoptó el candidato diagnóstico de
patrones como padre de este entrenamiento.

La etapa conserva un máximo de 2.048 actualizaciones y 1.800 segundos
cooperativos, con cuatro hilos CPU, 6 GiB de RAM y 2 GiB de corrida. Después
está prevista la retención conversacional existente y la generación de
desarrollo. Esos pasos posteriores no se cuentan como ejecutados en este corte.
La descarga y las actualizaciones no acreditan hechos aprendidos ni conversación
competente. El proceso depende de que el entorno continúe disponible.

El nuevo transporte es una recuperación optativa; **el supervisor vivo sigue
utilizando su configuración y recolector original fijados**. No se modificaron
sus fuentes ni se detuvo su trabajador. En una futura configuración se puede
seleccionar este helper y fijar su hash, con una ruta nueva de recibos para cada
invocación. No se presenta ese cambio futuro como aplicado al supervisor actual.

## Uso

Con la preparación previa del colector existente y sus rutas configuradas:

```bash
python examples/collect_learning_wiki_recovery.py \
  --recipe examples/learning-repair-20261013/collector-v2/recipe.json \
  --inbox-dir private/learning-repair-20261013/loop-inbox \
  --supervisor-state private/learning-repair-20261013/continual/state.json \
  --receipt-dir private/recovery-requests-new
```

El helper sólo colecta cuando el supervisor espera datos y existe un SFT propio
cerrado. Una ruta de recibos existente se conserva y se rechaza su reutilización.
La receta histórica requiere su seed, anclas, configuraciones y estado del
colector; copiarla a un checkout nuevo no inicia ni reconstruye esos archivos.
No lo lances para duplicar una colección activa.

## Evidencia

[Observación de recuperación e inicio de entrenamiento](recovery-observation.json),
[diagnóstico HTTP real](http-probes.json),
[integridad de caché](cache-integrity.json),
[revisión independiente de IA](independent-transport-review.json),
[verificación completa de datos y trabajador](verification.json) y los treinta
archivos en `successful-request-receipts` conservan entradas, hashes y respuestas
HTTP. Los fallos anteriores y sus logs permanecen en las rutas privadas de
recuperación. La revisión corresponde a un agente del proyecto, no a un auditor
humano independiente.

## Corrección de una prueba inestable detectada por CI

La primera ejecución completa de CI de la rama encontró un fallo en
`test_elapsed_handler_budget_prevents_verifier_and_keeps_actual_result`; la
ejecución del PR del mismo commit había pasado. La prueba asignaba diez
milisegundos desde la creación del encargo y asumía que el handler comenzaba
antes. La latencia de SQLite/fsync y del servidor pudo consumir ese tiempo
previamente; el controlador rechazó ejecutar la herramienta al vencer su límite.

Se conservan el [fallo original](historical-ci-failure-summary.txt), el límite
y todas las comprobaciones. El test ahora controla el reloj sólo dentro de
ese caso y avanza veinte milisegundos durante el handler, que sigue escribiendo
el archivo real. Verifica que no se llama al verificador después del vencimiento
y que se conserva el resultado y su hash. El código y los límites reales del
controlador no cambian. Sus [25 pruebas pasan](controller-clock-check.log).
