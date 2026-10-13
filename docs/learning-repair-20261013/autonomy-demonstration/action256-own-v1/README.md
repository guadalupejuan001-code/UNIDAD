# Acciones tras 256 updates: JSON válido en una propuesta, trabajo fallido

El modelo propio entrenado
`unidad-conversation-e202eafc8f11bcda7587db496c6e7ac1670efbd612685c465658ae2f07bd54fa`
se ejecutó por la API real con las mismas tres herramientas del baseline de
contexto 2.048. El prompt de 732 tokens cabe sin descarte. Produjo exactamente:

```json
{"action":"tool","tool":"inspect_file","arguments":{"filename":"k.txt"},"reason":"Medir"}
```

Es una acción JSON válida, de 63 tokens y final EOS. El nombre `k.txt` **no se
corrigió**: los archivos autorizados eran `dos.txt` y `uno.txt`. El controlador
terminó `blocked_policy` antes de llamar al handler, con un intento/fallo y sin
informe del modelo. Frente al JSON mal formado del baseline, esta única muestra
observa mejora parcial de formato; no demuestra selección correcta de archivos,
planificación fiable ni trabajo completado.

En la tarea mínima, prompt de 336 tokens, el texto real fue:

```text
{"tool":"inspect_file","arguments":{"filename":"inspect_file","filename":"inspect_file","filename":"inspect_file","},"reason":"Medir"}
```

Termina por EOS con 91 tokens, pero es JSON inválido y la tarea queda
`blocked_model`, sin herramientas ejecutadas. No hubo fallback ni edición de
los argumentos. `planningAccepted=false` en ambos intentos.

El [evidence.json original](evidence.json) tiene SHA
`ef2fa9da8ece2b9550cf59521a8369ae46ea21cedae2fff82eac19b8d7270789`.
Se copiaron exactamente cuatro archivos: evidencia, dos entradas ficticias y
el [informe de la fixture declarada](reports/informe.json). Ese informe procede
de acciones de prueba, no del modelo; `modelo.json` no existe. Los originales
conservan sus hashes, y la cancelación de la fixture tampoco guardó su informe.

El [recibo de copia](publication-integrity.json) conserva origen y hashes. Las
[huellas de fuente](source-fingerprints.json) extraen identidad de modelo,
política, prompt y propuesta de los registros reales y verifican el artefacto
publicado; no atribuyen al código una huella de ejecución que el registro no
capturó. No se copiaron SQLite, estado de tareas ni registros de memoria.
La publicación no ejecutó inferencia o entrenamiento.
