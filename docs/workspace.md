# Encargos, intentos y correcciones locales

El espacio guarda encargos en texto libre, el contexto que proporcionas, intentos realizados y tu aprobación o rechazo explícitos. No clasifica tu personalidad ni deduce preferencias. Registrar un encargo no lo ejecuta: este módulo no navega, abre referencias, ejecuta comandos ni fabrica respuestas.

```sh
unidad init --workspace private/default
unidad task add --workspace private/default --goal "Mi encargo" --context "Mis datos y condiciones"
unidad task attempt --workspace private/default --id TASK_ID --text "Mi intento realizado" --evidence "Mi referencia"
unidad task feedback --workspace private/default --id TASK_ID --attempt ATTEMPT_ID --text "Mi corrección o motivo" --accepted
unidad task list --workspace private/default
unidad task show --workspace private/default --id TASK_ID
unidad task export --workspace private/default --output private/corpus-1.txt
```

Sustituye `TASK_ID` y `ATTEMPT_ID` por los identificadores devueltos al registrar cada elemento. `--evidence` se puede repetir: conserva referencias declaradas por ti, sin verificar su contenido. `--rejected` registra un rechazo. La última valoración de cada intento decide si se puede exportar; un intento sin valoración queda excluido.

`task list` devuelve el ID, encargo, fecha de creación y conteos de intentos y aprobaciones actuales. Conserva la privacidad y el número de eventos del espacio; omite contexto, respuestas, referencias y comentarios. `task show --id` conserva el historial completo del encargo. Ambas consultas usan la misma lectura validada del registro. Ninguna ejecuta los encargos.

La API `search_tasks(path, query, limit=10)` y la operación de Brain
[`memory.task.search`](brain.md#memoria-entre-cuerpos) recuperan encargos por
palabras completas Unicode, sin distinguir mayúsculas y conservando los acentos.
NFC y NFD son equivalentes por palabra, mientras que `café` y `cafe` siguen
siendo distintos. Las marcas combinantes continúan la palabra original; sólo se
normaliza la comparación, conservando texto, offsets y SHA originales.
Todas las palabras deben aparecer en el mismo encargo: se consultan objetivo,
contexto, cuerpos de intentos actualmente aceptados y sólo la última corrección
de cada intento. El rechazo vigente se etiqueta como `accepted: false`; no se
recuperan cuerpos rechazados ni comentarios anteriores. La consulta tiene un
máximo de 512 bytes UTF-8 y 16 palabras distintas, con un límite de 1 a 20 tareas
(10 por defecto). Puede devolver cero coincidencias; los resultados se ordenan
por creación descendente e ID. Los extractos de hasta
160 caracteres incluyen IDs de tarea/intento/evento, campo, SHA del texto y
offsets originales verificables mediante `show`. Los offsets cuentan caracteres
Python y su final es exclusivo. Se informa el total de tareas coincidentes y si
la salida se ha acotado; no se copia todo el historial para buscar. La búsqueda
es léxica y explícita: no encuentra sinónimos, decide acciones ni añade recuerdos
automáticamente a una conversación.

Medición histórica del listado, anterior a la caché de replay, con un encargo ficticio, 25 intentos largos y 52 eventos: la salida JSON UTF-8 ocupó **300 bytes**, frente a **206.668 bytes** del historial completo, un **99,85 % menos**. Ambas salidas se midieron con `json.dumps(..., ensure_ascii=False)`. Esa reducción corresponde a los datos copiados y enviados; no mide el tiempo de la búsqueda ni de la caché actual. La lectura y el hash completos se mantienen, aunque la reconstrucción validada puede reutilizarse.

El corpus contiene únicamente el encargo, contexto, intento y último comentario de ejemplos actualmente aceptados. Su manifest, `corpus-1.txt.manifest.json`, conserva los IDs de los eventos originales, las referencias de evidencia y los SHA-256 del corpus y del historial usados. Sin intentos aceptados, no se crea ningún archivo. Una exportación requiere nombres nuevos y conserva archivos existentes, incluido el historial y posibles archivos de evidencia.

Cada ejemplo aceptado es un documento completo separado por una línea `---`, compatible con el lector del modelo de lenguaje. Una línea literal `---` dentro de tus textos se exporta como `\---` para conservar el ejemplo en un único documento; el texto original permanece intacto en el historial. El entrenamiento exige al menos tres documentos para separar entrenamiento, validación y prueba. La exportación conserva el número real de ejemplos: no inventa ni repite registros para alcanzar ese mínimo.

Aceptar o corregir un intento **no entrena parámetros**. Exportar tampoco los cambia. El corpus UTF-8 se puede suministrar a un entrenamiento explícito posterior del modelo de lenguaje. Un rechazo posterior afecta las siguientes exportaciones; no borra corpus previamente exportados ni modifica pesos ya entrenados.

Los datos permanecen locales y privados: usa `private/`, excluido de Git. Los registros son texto local sin cifrado. Se intenta establecer permisos Unix `700` para directorios creados y `600` para el historial, lock, corpus y manifest. En sistemas sin estos permisos, la protección depende del sistema de archivos y de tu cuenta. El módulo no envía datos por red. Un archivo exportado a otra ubicación sigue requiriendo el mismo cuidado al compartirlo.

`events.jsonl` se escribe agregando eventos, sin editar los anteriores, y sigue
siendo la autoridad. Cada operación lee el historial completo y calcula su
SHA-256 bajo el lock. Una caché pequeña en memoria reutiliza la reconstrucción
validada sólo cuando coincide el hash de todos los bytes actuales:
hasta cuatro espacios con historiales de hasta 2 MiB cada uno. No confía sólo en
fecha de modificación o tamaño. Una escritura invalida la entrada antes de
agregar el evento y la renueva únicamente después de `fsync`; la caché no se
guarda en disco y los resultados públicos quedan separados de su estado interno.
Ante un cambio del historial, o por encima del límite de caché, se reconstruyen
y validan esquema, identidades, asociación de intentos con encargos, orden
temporal y tipos JSON. Se rechazan claves duplicadas, números no finitos,
timestamps entre comillas y registros dañados; el archivo original se conserva
en lugar de reiniciarlo silenciosamente. El hash completo sigue costando tiempo
lineal en los bytes del registro; esta caché no resuelve historiales ilimitados.
Los escritores se coordinan con un lock de proceso en Unix y un lock entre hilos;
fuera de Unix sólo se garantiza coordinación entre hilos de un mismo proceso.

Cada campo de texto permite hasta 65.536 bytes UTF-8. Un intento admite hasta 16 referencias de evidencia de 8.192 bytes cada una; cada evento completo permite hasta 262.144 bytes. El historial inicial es una implementación sencilla que se reconstruye en memoria; un registro muy grande requerirá una evolución explícita de almacenamiento.

La API Python expone `init_workspace(path)`, `add_task(path, goal, context="")`, `record_attempt(path, task_id, text, evidence=None)`, `feedback(path, task_id, attempt_id, text, accepted)`, `list_tasks(path)`, `search_tasks(path, query, limit=10)`, `show(path, task_id=None)`, `status(path)` y `export_training(path, destination)`. `evidence` es una lista de referencias textuales; `accepted` debe ser un booleano explícito. Los resultados son objetos serializables a JSON. `add_task` y `record_attempt` devuelven su identificador en `id`. `show(path)` mantiene su devolución del historial completo para consumidores existentes.

## Medir la memoria local

El [informe medido](unidad-memory-optimization-v1.json) compara tres pares
alternados de procesos con 2.001 eventos ficticios: estado **20,5 → 0,7 ms**,
listado **20,5 → 0,8 ms**, un encargo **19,6 → 0,7 ms**, búsqueda
**50,3 → 7,8 ms** y agregar una corrección con `fsync` **20,8 → 1,0 ms**.
Cada cifra es la mediana de tres medianas, con calentamiento previo. La carga de
fondo no fue constante; no se midieron primera lectura, RAM máxima, historiales
por encima de 2 MiB, almacenamiento remoto ni cargas de producción.

La versión anterior carecía de API de búsqueda. Para comparar, el benchmark
aplica el mismo helper léxico actual al historial completo separado devuelto por
el `show` anterior, frente al `search_tasks` actual. Comprueba resultados JSON
idénticos, bytes del historial intactos al leer y rechazo vigente tras escribir.
Así se mide el efecto combinado de reutilizar el replay y evitar la copia del
historial completo para buscar; no se atribuye el ahorro a un único componente.

Desde la raíz del repositorio, con Python 3.11 o posterior:

```bash
memory_bench_dir="$(mktemp -d)"
git show 18b5b0a378f38c0c19d319c61bd95d262d4d19cb:src/ia_propia/workspace.py > "$memory_bench_dir/workspace-before.py"
python examples/benchmark_memory.py --source "$memory_bench_dir/workspace-before.py" --retrieval-helper src/ia_propia/memory_search.py --output "$memory_bench_dir/antes-1.json"
python examples/benchmark_memory.py --source src/ia_propia/workspace.py --retrieval-helper src/ia_propia/memory_search.py --output "$memory_bench_dir/despues-1.json"
```

Repite los dos últimos comandos con nombres nuevos (`antes-2.json`,
`despues-2.json`, etc.) para comparar varios pares. Las fuentes deben ser archivos
locales de confianza: el script ejecuta el módulo elegido y registra sus hashes.
Las salidas requieren rutas nuevas y no sobrescriben archivos. El benchmark
crea y elimina su propio historial ficticio; no necesita Torch ni tus encargos.
