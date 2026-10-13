La auditoría comprobó al supervisor **22557 / nacimiento 2592201** y al
entrenador CONTROL **23028 / nacimiento 2677279** activos. El recibo JSON guarda
la hora, los hashes y el estado observado: CONTROL llevaba **3647 / 4096**
pasos. Las fuentes nativas congeladas y los archivos fijados por el colector
coincidían. La inspección y la preparación no iniciaron, detuvieron ni señalaron
procesos; conservaron la configuración y el historial LIVE.

Los tres ciclos siguientes están **aprobados en cola, todavía no admitidos**.
Después de finalizar/exportar CONTROL, el supervisor genera sus 40 turnos DEV
en un hilo de CPU antes de admitir ACTION. Los nombres ordenados y los padres
explícitos establecen esta secuencia:

| Ciclo | Etapas y padre | Límites nativos por etapa |
| --- | --- | --- |
| ACTION | `action2048`, desde el artefacto propio ba70; después DEV40 | 256 actualizaciones o 600 segundos acumulados; 20 épocas |
| wide LATAM v3 | `language-1` desde ACTION, `language-2` desde la primera, `natural-action-retention` desde la segunda; después DEV40 | Cada causal: 2048 actualizaciones / 1800 segundos / 2 épocas. Repaso: 1024 / 1800 / 20 |
| Tatoeba siguiente | `language` desde el repaso wide, luego `natural-action-retention`; después DEV40 | Causal: 2048 / 1800 / 2. Repaso: 1024 / 1800 / 20 |

Son techos, no pasos garantizados: la paciencia, el tiempo o los recursos pueden
finalizar antes. Todas estas etapas usan cuatro hilos, BF16, lote lógico 2,
microlote 1, RSS cooperativo de 6144 MiB y disco de corrida de 2048 MiB. DEV usa
un hilo, 300 segundos, RSS 6144 MiB y salida de 512 MiB. Los turnos DEV se
generan mediante `ConversationEngine`; generar, revisar y aceptar siguen siendo
operaciones distintas. El repaso natural/acciones reutiliza datos conocidos y
no se contabiliza como conocimiento nuevo.

Collector V2 estaba habilitado e inactivo, sin hijo registrado. La cola aprobada
tiene prioridad. El colector sólo puede lanzarse cuando el bucle publica
`waiting_for_data`, después del intervalo de 3600 segundos y de comprobar sus
pines y el espacio. Tiene límites de 1200 segundos de proceso; su receta fija
80 páginas, 900 segundos de colección, 512 MiB de red, 24 MiB de corpus y 2 GiB
de disco. LIVE tiene `maxCycles: null`: espera mientras exista el entorno; las
etapas y las invocaciones sí son finitas.

No hubo discrepancias presentes de aprobación, corpus, configuración o
procedencia. Los padres de las etapas futuras aún no están materializados:
eso es esperado. Se comprobarán sus recibos y hashes al admitir/lanzar cada
etapa. Una falla de ACTION puede impedir admitir wide; una falla de wide puede
impedir Tatoeba. Las entradas inválidas se registran una vez por el hash exacto
de manifiesto y aprobación en `rejectedCycleHashes`, y otras entradas válidas
pueden continuar. El espacio requerido para lanzar es el techo de disco más
512 MiB. El JSON conserva las condiciones y los archivos de las colas.

La configuración pública `configs/continual-live-collector-v2.json` es una
captura de esta instancia: contiene `/workspace/UNIDAD`, un corpus inicial
privado y el PID histórico de adopción 9540. **No sirve para arrancar un clon
nuevo.** El helper
[bootstrap_learning_continuation.py](../../examples/bootstrap_learning_continuation.py)
prepara un experimento nuevo desde datos, configuraciones y artefactos públicos.
Sólo toma del archivo histórico el hook DEV fijado por hash. Calcula la raíz del
clon y reemplaza el ejecutable Python del hook por el intérprete actual.

Para reproducir un experimento nuevo, usa Linux y Python 3.11 o posterior en un
clon que contenga este helper y todos sus archivos públicos fijados. No necesita
modelos externos ni una descarga implícita de datos. Instala las dependencias
conversacionales declaradas y ejecuta desde la raíz del clon:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[conversation]'
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python examples/bootstrap_learning_continuation.py --output-root private/continuations/repro-v1 --dry-run
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python examples/bootstrap_learning_continuation.py --output-root private/continuations/repro-v1
.venv/bin/python examples/continual_training.py status --config private/continuations/repro-v1/config.json
```

El modo seco verifica todos los hashes, corpus reales, particiones y
configuraciones con el preflight nativo y no crea archivos. La publicación
posterior crea exclusivamente el directorio nuevo: `config.json`,
`state/state.json`, las dos colas posteriores con sus aprobaciones exactas y
`preparation-receipt.json`. Una renombración Linux `RENAME_NOREPLACE` publica el
directorio completo; un destino existente, un enlace simbólico o una carrera
para crearlo provocan rechazo, sin sobreescribirlo. La raíz nueva debe ser
`private/continuations/<nombre>`. No se permite apuntar al estado LIVE.

El primer ciclo nuevo es ACTION desde
`models/learning-repair-20261013/autonomy/context2048-untrained-parent1024`, modelo
propio **ba70fff…**. Hereda pesos propios entrenados; su extensión de contexto
512 → 2048 sigue sin entrenar, con cero pasos de optimizador desde la extensión.
El artefacto no incluye Adam. El helper no presenta esa extensión como capacidad
aprendida. Wide y Tatoeba siguen después con padres futuros explícitos; el
preflight no inventa recibos ni artefactos para validarlos.

El estado nuevo empieza sin resultados, pasos, presupuestos consumidos, PID ni
adopción históricos. Tiene `maxCycles: 3`, DEV40 en un hilo y el colector
deshabilitado. Para habilitar colección posteriormente, el anfitrión debe fijar
las condiciones de uso, el snapshot de origen, el cursor histórico, la semilla
de deduplicación y la caché cruda verificada. El helper no crea una historia de
colección ficticia. Conserva la distinción entre nuevo texto causal y repaso de
conversación/acciones.

Los comandos siguientes son la ejecución explícita de ese **nuevo** experimento
y su solicitud de parada. No se ejecutaron durante esta auditoría. El supervisor
usa un bloqueo por workspace: la reproducción se inicia en un clon separado con
los recursos disponibles, sin competir con LIVE por ese bloqueo.

```bash
.venv/bin/python examples/continual_training.py start --config private/continuations/repro-v1/config.json
.venv/bin/python examples/continual_training.py status --config private/continuations/repro-v1/config.json
.venv/bin/python examples/continual_training.py stop --config private/continuations/repro-v1/config.json
```

`stop` escribe una solicitud del supervisor. Para el entrenador activo, el
supervisor comprueba fuentes e identidad y publica una solicitud ligada al
propietario, runner, PID y nacimiento. El entrenador guarda en un límite
completo del optimizador; los checkpoints y efectos completados permanecen.
El hijo DEV puede recibir SIGTERM y después SIGKILL a los diez segundos, sólo
para su identidad propia. Esto no es una cancelación con reversión.

La reanudación automática del entrenador existe para `invocation_time_budget`
y para `stale` con checkpoint, con hasta dos reinicios por caída. Conserva Adam,
scheduler, RNG, corpus, padres, fuentes y límites acumulados. Una parada manual
queda en `paused_native`; no hay un comando genérico `resume` del supervisor.
El marcador de parada del supervisor permanece hasta que el anfitrión lo revise
y archive. El runner nativo admite `--resume` con los mismos argumentos y sólo
limpia una parada propia válida; no reabre una etapa que consumió su límite.
Después de una parada manual se debe comprobar el checkpoint y la identidad
antes de coordinar esa reanudación. Repetir `start` sin tratar el marcador no
reanuda el entrenamiento.

Para **reanudar exactamente LIVE** hacen falta el estado y la configuración
privados originales, `stage-design.json`, los checkpoints nativos con
optimizador/scheduler/RNG, los artefactos padres y los mismos bytes de datos y
fuentes. Los pesos exportados públicos permiten iniciar otro optimizador;
no reconstruyen la reanudación exacta. Un clon nuevo no debe copiar PID antiguos
ni resetear presupuestos cerrados. El helper reproduce la receta futura como
experimento nuevo, sin prometer pesos idénticos entre hardware ni una mejora
semántica.

La comprobación real preparó
`private/continuations/portable-reproduction-audit-v1` en **8,016 segundos** y
cargó su configuración mediante el CLI común. No creó `runs` ni `instance.json`;
`trainingStarted` y `supervisorAlive` eran falsos. Cinco regresiones pasaron en
0,026 segundos con un hilo; cubren modo seco, datos inválidos de una etapa
posterior, estado limpio, aprobaciones exactas y publicación sin reemplazo.
El [recibo JSON](loop-runtime-reproducibility-audit.json) conserva los hashes,
el estado LIVE observado y el recibo de preparación nuevo.
