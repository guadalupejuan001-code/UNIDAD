# Unidad: generalización de redacción y prueba propia

12 de octubre de 2026. Candidato experimental: **varied-s2-last**, paso
448 de esta etapa. No se promociona como
asistente competente ni cambia el modelo predeterminado.

En los 40 turnos conocidos, fallidas pasó de 28 a 15
(46.4% de reducción); completas pasó de 7 a 13,
críticos de 6 a 6 y diálogos completos de 0 a
2/10. Son casos de desarrollo usados para seleccionar.
En 40 turnos nuevos, C7 obtuvo C1/P4/F35 y el candidato
C3/P5/F32; los críticos fueron
9 y 15, respectivamente.
La reducción nueva fue 8.6%. No confundir ambos resultados.

## Qué se diagnosticó y cambió

Con el checkpoint propio anterior, ocho primeras preguntas reformuladas
pasaron de 7/8 coincidencias exactas a 0/8. En un mundo de validación con la
redacción practicada, hubo 43/48 coincidencias con historia redactada y también
43/48 con respuestas anteriores realmente generadas. En esa muestra, el
historial propio no explicaba la caída; la redacción desconocida sí la aislaba.
Exactitud literal es un diagnóstico, no una evaluación semántica completa.

Se probó beam de ancho 8 sin respuestas objetivo en inferencia: mejoró una
coincidencia literal de ocho, pero perdió una respuesta antes semánticamente
válida. Se descartó y no se incorporó al motor propuesto.

Se creó una etapa acotada desde `pointer-v1-best`, descendiente propio de C7:
5.120 prefijos nuevos de 128 mundos ficticios, diez familias y cuatro turnos,
más 512 registros exclusivamente TRAIN de repaso. Las 640 validaciones tienen
otros ocho mundos y dos paneles: redacción vista y una formulación reservada.
Las familias de habilidades se comparten entre TRAIN y esa validación; ésta no
demuestra transferencia a habilidades independientes. Se conserva el corpus
anterior y no se trasladó su VALIDATION a TRAIN. No se usaron chats personales
ni pesos externos. Los datos fueron redactados con conocimiento del diagnóstico;
la prueba conocida no se presenta como test final.

Se mantuvieron arquitectura, tokenizador, pérdida, cabeza de copia y generación.
El optimizador empezó nuevo para esta etapa y se reanudó, junto con su estado,
entre segmentos. Límite: 1.200 segundos nativos o 1.024 pasos, cuatro hilos CPU,
BF16 en entrenamiento, inferencia FP32. Fin observado: paso 679,
903.801 segundos, motivo nativo `invocation_time_budget`.
Se cerró la etapa después del tercer segmento al observar regresión en errores
críticos; no se consumió el cuarto tramo. `stage-closed.json` registra la decisión.
Los presupuestos antiguos permanecen cerrados.

Se evaluó después una única media aritmética de los pesos de 448 y 679 pasos,
con coeficientes 0,5 y 0,5 fijados antes de generarla. Empató C13/P12/F15 y seis
críticos, pero bajó de dos a un diálogo completo y de 6/40 a 3/40 coincidencias
en validación de redacción. Se descartó. La idea se inspira en
[promediado de pesos](https://arxiv.org/abs/1803.05407); no reproduce los
experimentos SGD/SWA de ese trabajo ni supone que su resultado se transfiera
a este pequeño Transformer entrenado con AdamW.

## Comparación conocida

Generación determinista: temperatura 0, semilla 42, hasta 32 tokens, un hilo;
se conserva cada respuesta propia como historia. C: completa, P: parcial,
F: fallida. Las críticas son turnos, no suma de defectos por turno.

| Candidato | Paso nuevo | C | P | F | Críticos | Diálogos completos | Paráfrasis literal | Validación literal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C7, referencia | — | 7 | 5 | 28 | 6 | 0/10 | — | — |
| varied-s1-best | 128 | 10 | 5 | 25 | 10 | 1/10 | 5/8 | 4/40 |
| varied-s1-last | 225 | 10 | 10 | 20 | 5 | 1/10 | 2/8 | 4/40 |
| varied-s2-best | 384 | 12 | 9 | 19 | 7 | 1/10 | 3/8 | 6/40 |
| varied-s2-last | 448 | 13 | 12 | 15 | 6 | 2/10 | 3/8 | 6/40 |
| varied-s3-last | 679 | 14 | 11 | 15 | 7 | 2/10 | 4/8 | 2/40 |
| varied-s3-mean | media 448+679 | 13 | 12 | 15 | 6 | 1/10 | 4/8 | 3/40 |

Se eligió por las respuestas reales y el límite de críticos de la referencia,
no sólo por la menor pérdida. Las columnas de coincidencia literal no son
calificaciones semánticas. Las respuestas parciales tampoco son completas.
Las revisiones son de una IA del proyecto, sin cegamiento; no constituyen una
evaluación humana o independiente. Se publican las razones turno por turno.

## Comprobación nueva

`fresh-transfer.json` se fijó antes de entrenar. Reúne diez situaciones nuevas
(recipientes, asientos, préstamos, revisión de avisos, elección y motivo,
propiedades compartidas, entrega, confirmación, medida y papeles en un evento),
cuatro turnos cada una. Comparte operaciones elementales de recuerdo y
corrección; no garantiza que cada habilidad sea totalmente nueva. Se comprobó
que no coincidieran literalmente sus mensajes completos con TRAIN.

La identidad del candidato quedó fijada en `candidate-selection.json` antes
de generar esta prueba. No se ajustaron pesos ni se eligió otro candidato
usando esos resultados. Después de publicarlos, este conjunto deja de ser
una prueba sin exposición para trabajos futuros.

| Modelo | C | P | F | Críticos | Diálogos completos |
| --- | --- | --- | --- | --- | --- |
| C7 | 1 | 4 | 35 | 9 | 0/10 |
| varied-s2-last | 3 | 5 | 32 | 15 | 0/10 |

La aceptación del proyecto sigue exigiendo 36 C/40, ocho diálogos completos
y ningún fallo crítico o técnico. No se ha alcanzado. La meta de reducir F un
50% en desarrollo figura como `False` y en
esta comparación nueva como `False`; no son
una estimación universal de calidad.

## Pesos e instrucción de carga

Modelo exacto: `unidad-conversation-1c3b781451c01cd1a54d89730d631c83720a775f3f7771ae225af267fd47aad8`.

- Pesos SHA-256: `bb45a1cedb688d0f7c2f3d21cf0c6b146142febd75a85d8da62c5a81b738e986`.
- Tokenizador SHA-256: `fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d`.
- Corpus nuevo SHA-256: `add631f57bd07f4a5cac85ed30a226670a2701b4a41781b4a0db3a49859f3449`.
- Prueba nueva SHA-256: `9af622085a17e2915f3d967806203128be4632e2cdda55bab19731716f296701`.

El paquete `Unidad-Prueba-Propia.zip` conserva todos los pesos evaluados,
referencia, punto de partida, estado final nativo con optimizador y corpus.
Las generaciones, revisiones y scripts se conservan en este repositorio.
Descomprimir y ejecutar `python restaurar.py
DIRECTORIO_NUEVO`; el restaurador verifica hashes y no sobrescribe directorios.
Con Unidad y sus dependencias conversacionales instaladas, desde ese directorio:

```bash
python -m ia_propia conversation converse --model models/varied-s2-last \
  --temperature 0 --seed 42 --max-new-tokens 32
```

La CLI nativa cargó exactamente ese identificador y reprodujo las cuatro
respuestas del diálogo de citas de la evaluación; se conserva su transcripción.
`/reiniciar` limpia el historial y `/salir` termina. La prueba sólo cambia el
historial de esa sesión; no entrena parámetros.

## Motor y Hormiga X

La [optimización de proyección](unidad-final-position-projection-20261012.md)
reduce aproximadamente a la mitad el tiempo de cálculo del siguiente token
en las dos longitudes medidas. Conservó 40/40 respuestas de C7. Esa mejora
de tiempo es distinta de reducir fallos de conversación.

La [revisión de Hormiga X](unidad-hormiga-compatibility-20261012.md) verifica
que su importador actual sólo admite GGUF y su contrato de razonamiento necesita
capacidades aún no validadas en Unidad. No se implementó un adaptador ni se
ejecutó inferencia en Android en esta etapa.

La evidencia textual está en `docs/wording-stage-evidence-20261012/`. Los
scripts ejecutados conservan sus rutas históricas de la sesión; ese directorio
ofrece además `evaluate.py` con rutas de modelos/casos configurables.
Los resultados fallidos y el beam descartado se conservan; no se reetiquetan
como mejoras. No queda un trabajador de entrenamiento activo al entregar.
