# Métodos conversacionales: consulta de fuentes originales

Revisión realizada el **13 de octubre de 2026 UTC** por una IA del proyecto.
No es una evaluación humana independiente ni una medición de Unidad. El alcance
es metodológico: un modelo propio de aproximadamente siete millones de parámetros
entrenado en CPU. Esta revisión no modifica el candidato congelado, el corpus,
la evaluación ni el código.

Se consultaron exclusivamente las publicaciones originales solicitadas. Los
arXiv se fijan en **v1** para identificar la publicación inicial; las fichas
actuales pueden contener revisiones posteriores. Se descargaron los documentos,
se extrajo el texto completo de los PDF y se leyeron
el resumen y las secciones indicadas abajo. No se utilizaron resúmenes de terceros.
Las decisiones para Unidad son inferencias del proyecto, diferenciadas de lo
que cada publicación demuestra.

## Fuentes y alcance de la lectura

| Ref. | Publicación original y versión consultada | Secciones leídas para esta revisión |
| --- | --- | --- |
| F1 | Ouyang et al., *Training language models to follow instructions with human feedback*, arXiv:2203.02155v1, 2022-03-04. [Ficha](https://arxiv.org/abs/2203.02155v1), [PDF](https://arxiv.org/pdf/2203.02155v1). | Resumen; §3.1–3.5, método, datos, tareas y SFT; §4.1, discusión sobre distribución de uso y conjuntos públicos. |
| F2 | OpenAI, *Introducing ChatGPT*, artículo fechado 2022-11-30. [Publicación](https://openai.com/index/chatgpt/). | Introducción, **Methods** y **Limitations**. Se consultó la página web actual del artículo histórico, no una captura archivada de 2022. La propia página advierte que ChatGPT ha evolucionado desde entonces. |
| F3 | Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv:2212.08073v1, 2022-12-15. [Ficha](https://arxiv.org/abs/2212.08073v1), [PDF](https://arxiv.org/pdf/2212.08073v1). | Resumen; §1.2–1.3, crítica/revisión y RLAIF; §3.1–3.3, datos, ajuste supervisado y evaluación. |
| F4 | Dubey et al., *The Llama 3 Herd of Models*, arXiv:2407.21783v1, 2024-07-31. [Ficha](https://arxiv.org/abs/2407.21783v1), [PDF](https://arxiv.org/pdf/2407.21783v1). | §2; §3.1–3.2.1; §4.1–4.2.3, incluido formato, máscara SFT, DPO, tablas 6–7, mezcla y limpieza. |
| F5 | Gemini Team, *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context*. [PDF oficial](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf). | Resumen; §3–4, arquitectura, eficiencia e infraestructura; §9.3.2–9.3.3, SFT y RLHF. El PDF recibido tiene 154 páginas y metadatos `CreationDate`/`ModDate` **2024-10-30 15:27:48 UTC**; la URL no fija una revisión numerada. No se presenta como el PDF inaugural de febrero. |
| F6 | Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388v1, 2025-05-14; portada fechada 2025-05-15. [Ficha](https://arxiv.org/abs/2505.09388v1), [PDF](https://arxiv.org/pdf/2505.09388v1). | Resumen; §3.1–3.2, datos y preentrenamiento; §4 y §4.1–4.4, arranque supervisado, RL, mezcla de modos, datos y recompensas. |
| F7 | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948v1, 2025-01-22. [Ficha](https://arxiv.org/abs/2501.12948v1), [PDF](https://arxiv.org/pdf/2501.12948v1). | Resumen; §1–2.4, base, GRPO, recompensas, SFT y destilación; §3.2 y §4–5, resultados y limitaciones. |
| F8 | Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290v1, 2023-05-29. [Ficha](https://arxiv.org/abs/2305.18290v1), [PDF](https://arxiv.org/pdf/2305.18290v1). | Resumen; §3–7 y apéndices A–B, SFT, preferencias, pérdida DPO, fundamento, experimentos, coste y cuestiones abiertas. |
| F9 | Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556v1, 2022-03-29. [Ficha](https://arxiv.org/abs/2203.15556v1), [PDF](https://arxiv.org/pdf/2203.15556v1). | Resumen; §1 y §3–5, preentrenamiento, reparto de cálculo y límites; apéndices A–C y F, mezcla, extrapolación y cálculo de FLOPs. |

## Qué publican y qué puede adoptar Unidad

**F1 — InstructGPT.** Parte de GPT-3 ya preentrenado, recoge demostraciones
de comportamiento deseado, ajusta mediante SFT y después aprende un modelo
de recompensa a partir de comparaciones humanas para optimizar la política con
PPO. Las entradas incluyen tareas diversas y se deduplican; las particiones por
usuario evitan compartir usuarios entre entrenamiento y evaluación. Su análisis
advierte que rendir en conjuntos públicos de tareas no equivale a cubrir la
distribución de uso real. **Adoptar:** separar base lingüística y ajuste, cuidar
la procedencia y la diversidad de peticiones, y reservar familias independientes.
**No trasladar:** sus épocas, tasas, resultados de preferencia ni su receta PPO
como garantías para siete millones de parámetros en CPU. Unidad no dispone de
la base GPT-3 ni de su equipo de etiquetadores.

**F2 — ChatGPT de 2022.** Describe conversaciones supervisadas en las que
entrenadores humanos representaron ambos lados, con acceso a sugerencias del
modelo, mezcladas con ejemplos de InstructGPT convertidos a diálogo. Las
comparaciones de respuestas se emplearon después para RLHF. Las limitaciones
publicadas incluyen respuestas falsas pero plausibles, sensibilidad a la
formulación, frases demasiado repetidas y preferencias que favorecen verbosidad.
**Adoptar:** diálogos de varios turnos que enseñen continuidad, correcciones y
peticiones actuales; controlar la repetición y la longitud de las demostraciones.
**No trasladar:** presentar autoría sintética del proyecto como conversación
humana, equiparar una revisión de IA a las preferencias humanas descritas, o
atribuir esta receta histórica a los productos actuales.

**F3 — Constitutional AI.** Su fase supervisada obtiene respuestas de un
asistente capaz, solicita críticas y revisiones y ajusta sobre respuestas
revisadas. La fase RLAIF compara pares mediante IA, aprende un modelo de
preferencias y optimiza con RL; las preferencias de utilidad siguen incluyendo
etiquetas humanas. El artículo relaciona la capacidad del evaluador con la
identificación de daños. **Adoptar:** una revisión editorial explícita que
compruebe hechos, roles y correcciones de las demostraciones antes del ajuste,
registrando que la revisión es de IA. **No trasladar:** llamar a esa revisión
editorial «RLAIF», asumir que un modelo pequeño puede autoevaluarse con fiabilidad,
o introducir un modelo de recompensa sin comprobar primero sus etiquetas.

**F4 — Llama 3.** Separa predicción del siguiente token sobre texto de SFT
y DPO posteriores. La mezcla y la limpieza del corpus se investigan con
experimentos; no son sólo un recuento de ejemplos. En §4.1.3, SFT usa entropía
cruzada sobre los tokens objetivo y enmascara los del prompt. La tabla 7 define
cada ejemplo como contexto —todos los turnos salvo el último— y respuesta final;
la tabla 6 describe descomponer conversaciones en ejemplos por turno. La sección
de calidad revisa patrones sintéticos repetidos y equilibra frases excesivas.
**Adoptar:** un objetivo de respuesta actual con historial completo como contexto,
enmascarar el contexto y comprobar separación de roles; variar formulación,
orden de información y composición de tareas, y controlar duplicados de familias.
**No trasladar:** los porcentajes de su mezcla, hiperparámetros, generación de
múltiples candidatos o DPO como opciones ya validadas para Unidad. Su resultado
de que DPO cuesta menos que PPO corresponde a sus modelos y recursos.

**F5 — Gemini 1.5.** El informe distingue Pro, una arquitectura MoE, de Flash,
un decoder optimizado y destilado de Pro. Describe entrenamiento en infraestructura
TPU distribuida, datos de varias modalidades, pares instrucción/respuesta y ajuste
posterior con preferencias humanas. La sección de SFT revisa respuestas y vigila
regresiones en otras métricas. **Adoptar:** medir conjuntamente calidad, tiempo,
memoria y regresiones al comparar intervenciones; conservar la distinción entre
datos base y pares conversacionales. **No trasladar:** arquitectura MoE, contexto
de millones de tokens, distilación ni latencias de sus API como una receta de
eficiencia para CPU. El informe no proporciona una reproducción completa de su
entrenamiento aplicable a Unidad.

**F6 — Qwen3.** Primero desarrolla una base lingüística y luego cambia la
mezcla y la longitud de contexto. Su informe controla calidad y diversidad de
datos con anotaciones y experimentos de modelos proxy. El posentrenamiento
separa arranque supervisado, razonamiento con verificadores, integración de
respuestas con y sin razonamiento y RL general; para modelos pequeños estudia
distilación desde profesores grandes. **Adoptar:** estudiar la distribución de
ejemplos y cambiar una hipótesis medible por experimento; favorecer respuestas
cotidianas breves cuando ésa sea la tarea. **No trasladar:** verificadores de
matemáticas a calidad conversacional abierta, bloques de razonamiento extensos,
las marcas de su chat template ni su ahorro de distilación al modelo propio de
Unidad. Ese ahorro compara procedimientos de Qwen bajo sus recursos.

**F7 — DeepSeek-R1.** R1-Zero comienza en **DeepSeek-V3-Base preentrenado**.
Su ausencia de SFT previo se refiere al posentrenamiento; no significa comenzar
desde pesos aleatorios. Estudia GRPO con grupos de respuestas y recompensas
de exactitud verificable y formato. R1 añade datos iniciales supervisados,
filtrado de respuestas y más SFT/RL. La destilación estudiada empieza en modelos
Qwen/Llama ya preentrenados; el menor publicado en v1 es de 1.5B parámetros.
Su §5 reconoce limitaciones en conversación multiturno, JSON, funciones, roles
y mezcla de idiomas. **Adoptar:** comprobar exactitud y legibilidad de las
demostraciones, y hacer la respuesta proporcional a la consulta. **No trasladar:**
«RL sin SFT» como aprendizaje de español desde cero, recompensas matemáticas
como juez de conversación abierta, o largas cadenas y etiquetas de razonamiento
como mecanismo que cree capacidad en Unidad. Los modelos destilados y los
experimentos de RL no verifican ese efecto en siete millones de parámetros.

**F8 — DPO.** Aprende de contextos y pares preferido/rechazado mediante una
pérdida que compara probabilidades frente a una política de referencia, normalmente
iniciada desde SFT. Evita un modelo explícito de recompensa y muestrear respuestas
durante el ajuste; no requiere el bucle RL de PPO. Conserva la necesidad de
preferencias. La equivalencia teórica usa supuestos sobre el modelo de preferencias
y la referencia; §7 deja abiertas la generalización fuera de distribución y la
sobreoptimización. **Adoptar:** distinguir el aprendizaje del idioma del ajuste
de preferencias y, si se estudia éste en una etapa futura, usar pares para el
mismo contexto con procedencia y calidad verificadas. **No trasladar:** DPO
como sustituto de una base/SFT útiles, sus hiperparámetros como valores óptimos
para CPU, ni una mejora de preferencia como prueba suficiente de conservación
de hechos o generalización multiturno.

**F9 — Chinchilla.** Investiga preentrenamiento autorregresivo y reparto de
cálculo entre parámetros y tokens; no proporciona una receta de SFT conversacional
o preferencias. Sus experimentos sustentan aumentar datos y capacidad conjuntamente
dentro del régimen estudiado. En §5 advierte sobre extrapolación y aclara que
los entrenamientos del análisis de escala usaron menos de una época. El apéndice A
sí registra repetición de algunos subcorpus en la mezcla final de Chinchilla:
ambas afirmaciones tienen ámbitos diferentes. **Adoptar:** repartir explícitamente
el presupuesto entre capacidad, texto diverso y duración, y medir curvas de
aprendizaje propias. **No trasladar:** el cociente aproximado de veinte tokens
por parámetro como requisito universal o garantía de conversación, ni convertir
su aproximación de FLOPs en horas de CPU sin medición. Unidad queda por debajo
de la escala de parámetros de esos experimentos y puede operar en un régimen
de repetición distinto.

## Decisiones metodológicas para un modelo propio en CPU

Estas decisiones son una adaptación del proyecto, no una receta demostrada
por las publicaciones para modelos de siete millones de parámetros:

1. **Base causal antes de especializar.** Conservar el linaje de pesos propios
   y separar adquisición de español general de SFT. Una colección pequeña de
   diálogos no sustituye una base lingüística. Evaluar por separado ambas etapas
   antes de atribuir a un optimizador un problema de idioma o generalización.
2. **SFT de la respuesta actual.** Usar el historial como contexto y supervisar
   la última respuesta de cada ejemplo. Para aprender varios turnos puede haber
   un ejemplo por prefijo que termine en assistant: cada respuesta es objetivo
   una vez y las anteriores pasan a contexto. Todos esos prefijos permanecen en
   la misma familia y partición. Esta elección debe comprobarse frente al modo
   anterior antes de adoptarla; aquí no se cambia el entrenador.
3. **Multiturno y distribución.** Cubrir peticiones actuales, hechos explícitos,
   identidad ficticia, relaciones, correcciones y aclaraciones con redacción
   variada. Separar conversaciones y familias evita atribuir generalización a
   sustituciones de nombres de una misma plantilla. Contabilizar ejemplos y
   tokens distintos además de pasadas y tokens procesados; repetir un corpus no
   aporta el mismo conocimiento que incorporar fuentes nuevas.
4. **Coste medido.** Presupuestar tiempo de CPU, tokens por segundo, memoria,
   longitud de contexto y calidad observada. Ejecutar comparaciones acotadas con
   recursos y condiciones equivalentes; las pérdidas de validación ayudan a
   diagnosticar pero no certifican conversación competente. Las leyes de escala
   orientan preguntas sobre datos y capacidad, no fijan una meta de tokens o
   horas universal para este modelo.
5. **Preferencias sólo con fundamento.** No implementar ahora RLHF, RLAIF ni DPO
   con preferencias emitidas por un juez débil sin validar. Una revisión editorial
   de IA permite detectar errores de un corpus, pero no establece una recompensa
   calibrada ni la fiabilidad de un optimizador de preferencias. Mantener SFT y
   comprobaciones de conversación como intervención identificable.

Ninguna fuente citada establece que estas técnicas conviertan por sí solas un
modelo propio de este tamaño y presupuesto en un asistente comparable a los
modelos de referencia. Las recetas publicadas se refieren a versiones históricas
identificadas; no documentan las recetas actuales de productos externos.

## Evidencia de descarga

Todos los accesos de esta tabla devolvieron **HTTP 200**. Se indica la hora
registrada de consulta UTC y el SHA256 de los bytes descargados, no del texto extraído.
Las copias de consulta y sus recibos quedaron en `/workspace/scratch/`; no son
corpus de entrenamiento ni artefactos del modelo.

| Ref. | Archivo consultado | Consulta UTC | Bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| F1 | PDF arXiv v1 | 2026-10-13 03:37:58 | 1.797.405 | `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98` |
| F2 | HTML de la publicación | 2026-10-13 03:37:58 | 565.677 | `9e984fa6389c6d177aefa891befc4d3f4d39075289dc772310e44eb646dc2540` |
| F3 | PDF arXiv v1 | 2026-10-13 03:37:58 | 2.088.111 | `9a456a07ad346e3372f9867d346f69f5b0f68b4c65f060aca0b8a13fa9d98e83` |
| F4 | PDF arXiv v1 | 2026-10-13 03:37:59 | 9.832.414 | `d7df3d6f537f025037f32ebe13198e3ea5500ea3e0967abdcceb80e31262d386` |
| F5 | PDF oficial, revisión identificada por hash | 2026-10-13 03:37:59 | 5.940.002 | `e9cabd1672ad0d32c2aa9c4e42d04dd55b519ea1945fbc2237bcbcb24b16add0` |
| F6 | PDF arXiv v1 | 2026-10-13 03:38:00 | 779.424 | `84a5e2b1fa04bb774bf12ae606d1e6d9dd2147ed2ebe78a4cfeb91ba380ffdd5` |
| F7 | PDF arXiv v1 | 2026-10-13 03:39:04 | 1.312.189 | `52d8ca3ac93e88cef9944e1fd03b0e04aec5954495a8250fb2fadf8fa20a4dad` |
| F8 | PDF arXiv v1 | 2026-10-13 03:39:04 | 1.286.755 | `304d57efe53317673ee5e93e1e1d990e3231fb4e24198aedd4829968e2c3cfbb` |
| F9 | PDF arXiv v1 | 2026-10-13 03:38:40 | 6.004.349 | `3fd3632a8ef48171bd25282990221d49535d75356192f068b3b2ebe08f2aedd4` |

También se descargaron estas fichas para comprobar identificador y revisión:

| Ref. | URL solicitada | Consulta UTC | SHA256 del HTML |
| --- | --- | --- | --- |
| F1 | https://arxiv.org/abs/2203.02155v1 | 2026-10-13 03:37:58 | `e4831ed1cc2d4e0ae05864ecabb81b29c493b4cb0ab51265783904fb09536145` |
| F3 | https://arxiv.org/abs/2212.08073v1 | 2026-10-13 03:37:58 | `2f175dac8902db65a5c0a7cd280109fa91e0d3c1d129e49edb9ab439edc3c97d` |
| F4 | https://arxiv.org/abs/2407.21783v1 | 2026-10-13 03:37:59 | `bd69b1a608451c4ccef7e46b2414fceab784fe35985f46783fad3ddb387b027c` |
| F6 | https://arxiv.org/abs/2505.09388v1 | 2026-10-13 03:38:00 | `0ea6b171fd31a303f340d0bdd192dedc76c68a7e709e8fb13d654ddc49a68e08` |
| F7 | https://arxiv.org/abs/2501.12948 | 2026-10-13 03:38:23 | `6dda5a280d3ea97c487505c8869d23a3831ad661e0be81cf74b862569bc66b84` |
| F8 | https://arxiv.org/abs/2305.18290 | 2026-10-13 03:38:23 | `c213a47bb5c67ad1983741bec43fc5e6ab252ae9737bd3e6c8daa2edbf5a3fe3` |
| F9 | https://arxiv.org/abs/2203.15556 | 2026-10-13 03:38:23 | `bc4809d5c487933ab6aed4ea233d1281b3e0b14ffba026cf441688f35d8ed146` |

Las fichas sin sufijo consultadas muestran DPO v3, DeepSeek-R1 v2 y Chinchilla v1.
Antes de fijar DPO v1 también se descargó
[DPO v3](https://arxiv.org/pdf/2305.18290v3), SHA256
`92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a`.
Ese PDF posterior **no se leyó ni se utilizó** para las conclusiones. Los hashes
de los PDF v1 identifican el contenido científico usado en esta revisión.
