# Conexión conversacional de NERVA y caché de generación

Intervención de software del 13 de octubre de 2026 iniciada sobre `main`
`63eede1649f7293bc8bf55cf603367b0bae853b5` y reconciliada con PR14/PR15 hasta
`85929202f7eb9c9a069dbaba3acc409c3f2195e3`. Conserva el avance del currículo de
patrones y todas las evidencias anteriores. No modifica pesos, tokenizador,
datos, checkpoints ni procesos de entrenamiento.

## Qué cambió

`ConversationEngine` ofrece dos selectores sobre los logits del mismo Transformer
propio. Torch conserva su muestreo anterior cuando los controles son neutrales.
NERVA ejecuta `conversation_decoder.nv` en un proceso Node por respuesta,
reutilizado entre tokens; no ejecuta el cálculo neuronal del Transformer.
El programa es fijo, con límites de entrada, decisiones y tiempo, y errores
explícitos sin cambiar de backend. El proceso y sus archivos se cierran al salir.

CLI, Brain/API, GNOSIS y evaluación comparten esta conexión. El anfitrión elige
el backend y la caché; el diálogo no puede elegir código, ejecutables o rutas.
Las respuestas registran la ejecución y mantienen la identidad de los pesos.
Las configuraciones GNOSIS anteriores siguen admitidas. Se conserva el nombre
SOVENGI, la selección de `reference-wiki1` de 15,77 M, su memoria, su API y el
bloqueo de inferencia incorporados por PR15. `sovengi chat` y `sovengi ask`
reciben los mismos controles y semilla por el contrato compartido.

La caché guarda claves y valores de atención dentro de una sola respuesta.
Reutiliza el prefijo sin volver a calcular cada token anterior. Al desplazarse
la ventana de contexto se reconstruye: las posiciones aprendidas del modelo
también cambian. Conserva la cabeza de copia, EOS, el bloqueo por modelo, los
RNG locales y el modo entrenamiento/evaluación anterior. No se guarda memoria
conversacional entre peticiones ni se añaden parámetros a los artefactos.
La caché consume memoria adicional para ahorrar cálculo; no se afirma reducir RAM.

Ambos selectores ofrecen top-k, top-p y penalización de repetición sobre la
respuesta en curso. Los valores por defecto son neutrales: 0, 1 y 1. La
penalización no se aplica a los hechos del usuario. No se escogió una política
de calidad usando la evaluación reservada. En muestreo, Torch y NERVA emplean
RNG distintos; igualdad de semilla no significa igualdad entre backends.

## Medición conservada

[Casos](benchmark-cases.json) y [resultado completo](benchmark-c7.json): diez
primeros turnos conocidos de desarrollo, tres repeticiones con orden rotado,
temperatura cero, semilla 42 y hasta 160 tokens. El modelo es C7, con
6.983.937 parámetros, cabeza de copia, vocabulario 8.192 y contexto 512.
Se incluyen las 90 generaciones reales, tiempos individuales, versiones,
identidad del modelo y hashes antes/después. No se leen respuestas de referencia
para decidir los tokens. El tiempo incluye la respuesta completa, arranque de
Node, intercambio de datos y cierre; excluye cargar el modelo.

| Recorrido | Mediana de las diez consultas | Relación con el anterior |
| --- | ---: | ---: |
| Torch sin caché, referencia | 12,723 s | 1,00 |
| Torch con caché | 4,368 s | 2,91 veces más rápido |
| NERVA con caché | 27,131 s | 2,13 veces más lento |

Todos los tokens y motivos de parada coincidieron con la referencia en cada
caso y repetición. Los hashes de configuración, tokenizador y pesos permanecieron
iguales. No hubo finales por presupuesto en esta muestra C7. Los tiempos varían
en el entorno compartido; el JSON conserva las tres mediciones completas, no
sólo la mediana. Son observaciones de esta CPU, no garantías universales.

**Decisión:** activar Torch con caché como predeterminado; mantener NERVA
conectado como selector opcional y verificable. Activar NERVA en esta medición
no aporta velocidad. Su utilidad implementada es poder ejecutar y comparar una
política de selección dentro del lenguaje propio, conservando el modelo.

Los resultados no equivalen a medir calidad conversacional. Por ejemplo,
`library_initial_deadline` sigue produciendo en C7 una respuesta repetitiva que
no recupera correctamente el plazo. Conservar sus tokens también conserva ese
error. Este C7 no es el candidato anterior a Tatoeba ni el posterior; sus
resultados no sustituyen aquella comparación A/C. Tampoco se midió aquí la
velocidad de los pesos actuales de patrones de 15,77 millones, ni Android/GPU.

La concordancia numérica de la caché se comprueba con tolerancias y las secuencias
medidas coinciden. Distinto tamaño de las operaciones puede alterar redondeos
cerca de empates en otros pesos o dispositivos; la ruta sin caché sigue disponible.

## Uso y reproducción

Después de instalar las dependencias conversacionales y Node.js 18+:

```sh
unidad conversation converse --model models/conversation-reference-c7 \
  --decoder-backend nerva --temperature 0 --max-new-tokens 160 --seed 42

unidad conversation converse --model models/conversation-reference-c7 \
  --decoder-backend torch --temperature 0 --max-new-tokens 160 --seed 42
```

Para el recorrido de referencia anterior, añadir `--no-cache`. Para NERVA en
GNOSIS, fijar `brain.conversation_decoder` en `"nerva"` en la configuración del
anfitrión. El archivo entregado usa `"torch"` y `conversation_cache: true`.
En Brain se usa `--conversation-decoder nerva`; no se envía ese selector como
campo de `conversation.reply`. Los controles de variedad sí son argumentos
validados de la petición, con valores neutrales por defecto.

```sh
PYTHONPATH=src python examples/benchmark_conversation_decoding.py \
  --model models/conversation-reference-c7 \
  --cases docs/nerva-conversation-20261013/benchmark-cases.json \
  --output /tmp/unidad-decoding-nueva-medicion.json --repeats 3 --torch-threads 1
```

El runner exige salidas nuevas y conserva un recibo parcial si se interrumpe.
No sobrescribir el benchmark entregado. Las pruebas pertinentes cubren logits
con y sin copia, desplazamiento de contexto, tokens, RNG, pesos, concurrencia,
limpieza por fallo/timeout, selección NERVA real y recorrido por Brain/GNOSIS.
El [recibo de verificación](verification.json) identifica 152 pruebas pertinentes
aprobadas y una comprobación posterior de 42 pruebas al reconciliar SOVENGI,
con módulos compartidos entre ambas baterías; no se suman como pruebas únicas.

## Qué falta para mejorar las respuestas

Esta entrega no demuestra una mejora semántica ni resuelve la regresión de
Tatoeba. Permite medirla con más control: fijar un checkpoint y comparar primero
el recorrido anterior, después sólo el selector y después sólo una penalización;
conservar las respuestas y medir completitud, hechos, correcciones, repetición
y EOS por separado. Evitar ajustar múltiples controles simultáneamente.

Para atribuir la regresión al entrenamiento, se necesitan los checkpoints
anterior, posterior a Tatoeba y posterior a retención bajo la misma generación,
incluyendo prefijos idénticos y diálogos con historia propia. Los controles de
patrones ya mostraron aprendizaje de ejemplos y escasa generalización; acelerar
la ejecución o filtrar tokens no acredita haber corregido esa limitación.
