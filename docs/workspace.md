# Encargos, intentos y correcciones locales

El espacio guarda encargos en texto libre, el contexto que proporcionas, intentos realizados y tu aprobación o rechazo explícitos. No clasifica tu personalidad ni deduce preferencias. Registrar un encargo no lo ejecuta: este módulo no navega, abre referencias, ejecuta comandos ni fabrica respuestas.

```sh
ia-propia init --workspace private/default
ia-propia task add --workspace private/default --goal "Mi encargo" --context "Mis datos y condiciones"
ia-propia task attempt --workspace private/default --id TASK_ID --text "Mi intento realizado" --evidence "Mi referencia"
ia-propia task feedback --workspace private/default --id TASK_ID --attempt ATTEMPT_ID --text "Mi corrección o motivo" --accepted
ia-propia task show --workspace private/default --id TASK_ID
ia-propia task export --workspace private/default --output private/corpus-1.txt
```

Sustituye `TASK_ID` y `ATTEMPT_ID` por los identificadores devueltos al registrar cada elemento. `--evidence` se puede repetir: conserva referencias declaradas por ti, sin verificar su contenido. `--rejected` registra un rechazo. La última valoración de cada intento decide si se puede exportar; un intento sin valoración queda excluido.

El corpus contiene únicamente el encargo, contexto, intento y último comentario de ejemplos actualmente aceptados. Su manifest, `corpus-1.txt.manifest.json`, conserva los IDs de los eventos originales, las referencias de evidencia y los SHA-256 del corpus y del historial usados. Sin intentos aceptados, no se crea ningún archivo. Una exportación requiere nombres nuevos y conserva archivos existentes, incluido el historial y posibles archivos de evidencia.

Cada ejemplo aceptado es un documento completo separado por una línea `---`, compatible con el lector del modelo de lenguaje. Una línea literal `---` dentro de tus textos se exporta como `\---` para conservar el ejemplo en un único documento; el texto original permanece intacto en el historial. El entrenamiento exige al menos tres documentos para separar entrenamiento, validación y prueba. La exportación conserva el número real de ejemplos: no inventa ni repite registros para alcanzar ese mínimo.

Aceptar o corregir un intento **no entrena parámetros**. Exportar tampoco los cambia. El corpus UTF-8 se puede suministrar a un entrenamiento explícito posterior del modelo de lenguaje. Un rechazo posterior afecta las siguientes exportaciones; no borra corpus previamente exportados ni modifica pesos ya entrenados.

Los datos permanecen locales y privados: usa `private/`, excluido de Git. Los registros son texto local sin cifrado. Se intenta establecer permisos Unix `700` para directorios creados y `600` para el historial, lock, corpus y manifest. En sistemas sin estos permisos, la protección depende del sistema de archivos y de tu cuenta. El módulo no envía datos por red. Un archivo exportado a otra ubicación sigue requiriendo el mismo cuidado al compartirlo.

`events.jsonl` se escribe agregando eventos, sin editar los anteriores. Cada operación reconstruye y valida su esquema, identidades, asociación de intentos con encargos, orden temporal y tipos JSON. Rechaza claves duplicadas, números no finitos, timestamps entre comillas y registros dañados; conserva el archivo original en lugar de reiniciarlo silenciosamente. Los escritores se coordinan con un lock de proceso en Unix y un lock entre hilos; fuera de Unix sólo se garantiza coordinación entre hilos de un mismo proceso.

Cada campo de texto permite hasta 65.536 bytes UTF-8. Un intento admite hasta 16 referencias de evidencia de 8.192 bytes cada una; cada evento completo permite hasta 262.144 bytes. El historial inicial es una implementación sencilla que se reconstruye en memoria; un registro muy grande requerirá una evolución explícita de almacenamiento.

La API Python expone `init_workspace(path)`, `add_task(path, goal, context="")`, `record_attempt(path, task_id, text, evidence=None)`, `feedback(path, task_id, attempt_id, text, accepted)`, `show(path, task_id=None)`, `status(path)` y `export_training(path, destination)`. `evidence` es una lista de referencias textuales; `accepted` debe ser un booleano explícito. Los resultados son objetos serializables a JSON. `add_task` y `record_attempt` devuelven su identificador en `id`.
