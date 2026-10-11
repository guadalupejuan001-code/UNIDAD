# Unidad: candidato conversacional uniforme B1

Artefacto **experimental**, sin aceptación de conversación básica ni promoción
como modelo predeterminado. Los cuatro archivos nativos conservan exactamente
los bytes del checkpoint seleccionado, sin cambiar pesos, configuración,
tokenizador o manifiesto.

Transformer propio de **6.967.296 parámetros**, BPE propio de **8.192 tokens** y
contexto de **512 tokens**: seis capas, dimensión 256 y ocho cabezas. Su procedencia
es pesos aleatorios propios → preentrenamiento 6000 → SFT 6000 → etapa uniforme
B1. B1 partió de [conversation-language-v7](../conversation-language-v7/README.md)
con Adam y scheduler nuevos, semilla 42, supervisión de la última respuesta del
asistente, media por token y muestreo uniforme sin reemplazo. Conservó el BPE.

B1 se cerró deliberadamente mediante parada cooperativa en el paso **2786** para
cambiar el modo de cálculo en una etapa posterior. Seleccionó el checkpoint
**2000** por validación, con NLL media por token **3.756617233475704**; la validación
inicial del mismo corpus y objetivo fue **4.050960822047885**. Es un cierre parcial:
no acredita fallo, estancamiento ni completar 24000 pasos. La pérdida de
validación no demuestra conversación competente.

En los mismos **40 turnos realmente generados**, la revisión primaria obtuvo
**2 completas, 6 parciales y 32 fallidas**; la cruzada, **2 completas, 4 parciales
y 34 fallidas**. Ambas registraron **0 diálogos completos**. La primaria señaló
**14 turnos críticos** y la cruzada **17**; los 14 de la primera están incluidos
en los 17 de la segunda. Se conservan los dos recuentos y los desacuerdos sin
afirmar una adjudicación. Las revisiones corresponden a asistentes de IA del
mismo proyecto, con identidad del modelo visible y sin panel humano externo.
Cada una se guardó antes de consultar las calificaciones de la otra. B1 no
superó la aceptación.

La generación terminó por EOS en 37 respuestas y por el límite de 32 tokens en
3; no se truncó el contexto de entrada. El
[informe de resultados](../../docs/unidad-uniform-coverage-b1-results.json)
conserva la evidencia y sus límites. Estos diálogos de desarrollo ya expuestos
no equivalen a una evaluación final independiente ni a competencia general.

El encoder materializó la partición TEST; no calculó su pérdida ni la usó para
selección o evaluación. No se incorporó el chat del creador ni se usaron pesos
preentrenados externos o una API generadora. No demuestra programación autónoma.

El [registro de fuentes](../../docs/unidad-spanish-language-base-v1.json) conserva
la procedencia, revisiones y declaraciones de licencia de Wikipedia y
OpenAssistant, con los diálogos ficticios identificados aparte. Las declaraciones
de las fuentes no certifican derechos de cada texto ni de los pesos entrenados.

Modelo: `unidad-conversation-737364a8c6f3a798aa4352fffed85ad7bec109ff26b0359510aa3bdab8e81ad6`.

SHA-256 del manifiesto: `67d77df08bbc27f06ed4d93fe6ed4fb0401c0ec2e4e4776b5771343f0119d4ad`.

SHA-256 de los pesos: `67a77ccc76bb51ad0b62828b8e557b4ed0a1a1a8fe45383e3c4334430d409155`.
