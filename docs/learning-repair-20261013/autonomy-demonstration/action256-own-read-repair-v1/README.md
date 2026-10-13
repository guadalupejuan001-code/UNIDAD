# Un reintento real de lectura: tarea todavía fallida

La API autenticada ejecutó el mismo modelo propio `e202eafc…`, sin cambiar
pesos, con `max_argument_repairs=1`. La primera propuesta real fue exactamente:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"k.txt"},"reason":"Medir"}
```

El controlador rechazó `k.txt` antes del handler y permitió otra propuesta,
sin editar la primera ni ampliar permisos. La segunda respuesta fue:

```text
{"action":"tool","tool":"save_report","arguments":{"filename":"save_report","save_report","arguments"filename":"save_report","save_report","save_report","arguments"filename":"j.json","files":["j.txt","json","json","
```

Es JSON mal formado. El resultado real es **blocked_model**, dos pasos, dos
fallos y una reparación de argumentos. No se ejecutaron efectos, no existe
informe, los originales conservan sus hashes y `fixtureFallback=false`.
La opción permite intentarlo de nuevo; este intento no demuestra recuperación
aprendida ni trabajo fiable.

El [evidence.json original](evidence.json) tiene SHA
`72886bdb689cde178b6134ed1a799f8dfd9d3620fdf7bb373cfa6b63d2c5fe91`.
Se copiaron tres archivos exactos: evidencia y dos entradas ficticias. El
[recibo](publication-integrity.json) conserva los hashes y las huellas de código
registradas en la ejecución. No se copiaron SQLite ni memoria; no hubo nuevas
llamadas de modelo o entrenamiento durante la publicación. La
[prueba anterior sin reparación](../action256-own-v1/README.md) permanece intacta.

Reproducción mediante el helper de API real, con carpeta nueva:

```sh
python3 examples/demo_autonomy_argument_repair.py --help
```

El helper utiliza directamente los pesos propios y el verificador independiente
del host, sin una fixture de propuestas como sustitución del modelo.
