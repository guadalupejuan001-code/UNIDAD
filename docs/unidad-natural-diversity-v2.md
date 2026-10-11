# Currículo sintético natural de Unidad v2

El banco `unidad-natural-diversity-v2` amplía las conversaciones breves en español
con hechos ficticios, correcciones y hechos independientes intercalados. Su
objetivo es aportar material para una etapa experimental; no acredita que Unidad
ya converse competentemente. El autor es otro asistente de desarrollo, sin
revisión humana atribuida.

`examples/build_natural_diversity_conversations.py` es un helper de autoría offline.
La inferencia no lo importa. No carga pesos, no usa otra IA para responder como
Unidad, ni usa el historial del creador o casos/resultados de evaluación para
inventar objetivos. Los datos personales anteriores permanecen separados.

El diseño contiene 120 mundos, con 12 flujos de cuatro o cinco intercambios por
mundo. Enseña el nombre de Unidad frente al nombre real y el apodo del usuario,
correcciones del nombre, lugar actual frente al anterior, tema actual frente al
anterior, colores de dos objetos distintos y cambios entre escuchar y proponer.
Los resúmenes conectan solamente nombre, lugar y tema compartidos explícitamente.
Los objetos incluyen vaso, camisa, cuaderno, plato, bolso, taza, silla, mesa,
botella y libro; artículos y colores concuerdan en género.

Cada evento tiene ocho redacciones comunes a entrenamiento, validación y test.
Los mundos usan el mismo conjunto de componentes léxicos. Los valores completos
de nombres, apodos, ciudades y proyectos son únicos y combinan una, dos o tres
palabras. Su longitud se distribuye independientemente del papel inicial o
corregido, y ningún valor completo aparece en dos atributos de mundos distintos.
Las ubicaciones ficticias se mezclan con nombres públicos de ciudades;
ninguna es una afirmación sobre el creador. Las aperturas expresan un dato natural
del mundo. Las preguntas sobre la identidad de Unidad aparecen en los turnos
2, 3, 4 y 5, evitando una apertura genérica que pueda cruzar particiones.

Las particiones se fijan antes de comprobar longitud y colisiones: 96 mundos de
entrenamiento, 12 de validación y 12 de test. Todas las versiones y prefijos de
cada mundo conservan su partición. Los 120 prefijos exactamente duplicados dentro
del mismo mundo se representan una sola vez. Los prefijos se entrenan únicamente
con la última respuesta del asistente; las respuestas anteriores son contexto.
No se trunca ni se descarta un diálogo para ajustar su longitud. El helper falla
si un diálogo completo supera el límite del contexto o si hay una colisión.

Los archivos privados congelados son:

- `private/conversation/natural-diversity-v2.jsonl`: 6.960 registros; 5.568 de
  entrenamiento, 696 de validación y 696 de test.
- `private/conversation/natural-diversity-v2.manifest.json`: diseño, procedencia,
  coberturas, longitudes, políticas y hashes.
- `private/conversation/natural-diversity-v2.bank.json`: banco literal de
  paráfrasis, flujos y componentes léxicos.

Los tres se publican con creación exclusiva y permisos 0600. Una ruta ya creada
nunca se sobrescribe. El JSONL tiene SHA-256
`60774864d95dfdf5d89c9b47fd435048bb6be1bc35ef512530355e90b21169d6`;
el banco tiene SHA-256
`0805278536b2a1f17f1ac802b313f67df1da0e9244c926a791d0d4439f163f66`.

Se conservó sin reajustar el tokenizador BPE de 2.048 entradas en
`private/conversation/snapshots/variable-flow-parent-before-optimization/tokenizer.json`,
SHA-256 `ee3b0f263ce97919f0a6b6ea8bd1329d1a008883116ee3a2decc9eb15f6fbb59`.
Los registros ocupan entre 52 y 204 tokens, incluyendo el sistema predeterminado
y EOS, dentro del límite de 257 tokens para contexto 256. No se excluyeron flujos.

La auditoría de colisiones leyó exclusivamente el corpus autorizado
`private/conversation/recovery-copy-flow-mix-v1.jsonl`, 42.462 registros y SHA-256
`c113e92958a76e4c94a0b11b78268134dbd8f496bbc817f252b0acad2db31d3f`.
Retuvo hashes normalizados de prompts completos y aperturas con sus particiones;
no retuvo respuestas objetivo ni métricas. Encontró cero cruces de partición en
ambas clases de prompt, entre mundos nuevos y contra ese corpus histórico. No
movió ninguna familia o partición anterior. Antes de incorporar otros corpus,
hay que repetir la auditoría contra todos los que participen en esa mezcla.

La verificación independiente se encuentra en
`tests/test_natural_diversity_conversations.py`. Sus hechos esperados proceden del
texto del usuario y de fixtures escritos aparte; no se calcula el resultado
esperado reproduciendo el estado o selector de respuestas del helper. Las pruebas
de datos, partición, máscaras y reproducibilidad no prueban la capacidad generativa
del modelo. La aceptación de conversación básica sigue siendo un paso separado.

La revisión v2 corrigió una asociación espuria del prototipo v1 entre longitud
del valor y su papel inicial/corregido, y cinco coincidencias literales de apodos
con lugares de otro mundo. Ambas se detectaron revisando la estructura del banco,
sin cargar un modelo ni leer resultados de evaluación. Los archivos privados v1
se conservan inmutables como prototipo no entrenado; también se preservó su helper
original en `private/conversation/natural-diversity-v1.generator.py`. La revisión
conserva los 120 índices de mundo, proporciones y 12 flujos; cambia los
valores ficticios, nunca la partición para arreglar una colisión.

La suite independiente aprobó 13 pruebas. Además se verificaron los 6.960
objetivos del JSONL congelado con el oráculo de texto del usuario, y los 6.960
registros con las máscaras del entrenador real: sólo última respuesta y EOS,
sin cargar el backend del modelo. Una segunda construcción, incluida la auditoría
histórica, reprodujo exactamente los bytes del JSONL y del manifiesto. Los hashes
del helper, pruebas y artefactos, junto con estas comprobaciones, están en
`private/conversation/natural-diversity-v2.qa.json`, también privado e inmutable.
