# Modelo pequeño de lenguaje propio

Esta base entrena una red recurrente desde pesos aleatorios para predecir el
siguiente carácter. No descarga pesos ni consulta una API. Es una implementación
real de entrenamiento y generación; su capacidad se limita a patrones de texto
del corpus. No demuestra comprensión, seguimiento de instrucciones ni trabajo
digital autónomo.

`examples/language-demo.txt` contiene diez textos en español escritos para esta
demostración. Son ejemplos inventados, no conversaciones ni datos de una persona
usuaria. Una línea `---` separa documentos. El lector elimina espacios exteriores
de cada documento y nunca crea una transición entre documentos.

## Entrenar y generar

Desde la raíz del proyecto, después de instalar sus dependencias:

```bash
PYTHONPATH=src python -m ia_propia.language train \
  --corpus examples/language-demo.txt \
  --output models/language-demo.json \
  --report models/language-demo-report.json

PYTHONPATH=src python -m ia_propia.language generate \
  --model models/language-demo.json \
  --prompt 'La red ' --length 160 --seed 42 --temperature 0.8
```

El entrenamiento usa NumPy, una capa recurrente `tanh`, retropropagación manual
a través del tiempo, Adam y recorte de la norma global del gradiente. Los valores
iniciales son aleatorios; los sesgos empiezan en cero. La configuración inicial
es pequeña: 24 unidades ocultas, fragmentos de 48 caracteres y hasta 20 épocas.
`--hidden-size`, `--sequence-length`, `--epochs`, `--learning-rate`, `--clip-norm`,
`--patience` y `--seed` permiten cambiar el experimento dentro de límites
explícitos. Un nuevo entrenamiento guarda un artefacto separado; no modifica un
modelo de producción ni aprende silenciosamente de una conversación.

La generación es reproducible con el mismo artefacto, prompt, temperatura y
seed. Temperatura cero elige el carácter con mayor puntuación. Si el prompt está
vacío se usa el carácter más frecuente del entrenamiento; la salida lo muestra
como `effectivePrompt`. El marcador desconocido no se emite. La generación
conserva estado durante el prompt y la continuación, pero no promete contexto
fiable más largo que los fragmentos entrenados.

## Evaluación y procedencia

Los documentos se dividen en bloques contiguos disjuntos: entrenamiento,
validación y prueba, por defecto 6/2/2. Documentos exactamente iguales en bloques
distintos se rechazan. El vocabulario contiene exclusivamente caracteres del
entrenamiento; caracteres nuevos en los otros bloques se evalúan como `UNK` y
su cantidad queda registrada. La línea base usa frecuencias de los caracteres
objetivo de entrenamiento con suavizado aditivo de uno.

Cada fragmento empieza con estado oculto cero. Dentro del fragmento se conservan
todas las transiciones y todos los gradientes. La evaluación informa pérdida
media por carácter siguiente en **nats**, cantidad de objetivos y perplejidad.
El checkpoint se elige sólo por la pérdida de validación; la prueba se evalúa
después y no modifica pesos, vocabulario, línea base ni selección. Un checkpoint
de época cero significa que ningún entrenamiento mejoró la validación: se guarda
ese resultado y no se presenta como aprendizaje exitoso.

El artefacto conserva tokenizer, configuración, pesos, seed, hashes de datos y
pesos, documentos y bloques utilizados, versiones Python/NumPy y métricas. Su
hash de contenido y su identidad se verifican al leerlo, junto con tipos,
dimensiones y valores finitos. Ese hash detecta cambios del archivo; no prueba
la autenticidad externa del corpus. El informe conserva la historia de pérdidas
y la comparación con la línea base, incluso si la red queda peor.

El corpus y los artefactos son archivos locales; el entrenamiento no cifra ni
anonimiza texto. Los pesos pueden memorizar fragmentos. Elegir un corpus propio
y entrenarlo son acciones explícitas: esta base no busca archivos privados ni
exporta información automáticamente.

El resultado en estos diez textos sólo comprueba el recorrido técnico y un
experimento pequeño. Un chatbot útil o un trabajador digital necesita datos,
arquitectura, herramientas y evaluación de tareas diferentes.

## Primera corrida guardada

Se entrenó una sola corrida con la configuración predeterminada: 2.903 parámetros,
47 tokens contando `UNK`, 13 épocas completadas y checkpoint elegido en la época 7.
El conjunto test tiene 630 objetivos; su pérdida fue **2,201882 nats**, frente a
**2,871445** del unigram de entrenamiento. La validación contiene un carácter
desconocido, registrado en el informe. El test no se usó para ajustar la corrida.

El [artefacto](../models/language-demo.json), [informe](../models/language-demo-report.json)
y [generación obtenida](language-demo-generation.json) se conservan. La generación
todavía produce muchas palabras sin sentido. Esta mejora estadística sobre una
demostración estrecha no acredita comprensión ni capacidad de trabajar.
