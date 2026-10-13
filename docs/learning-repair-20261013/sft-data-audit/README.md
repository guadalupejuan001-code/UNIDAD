# Auditoría de TRAIN del SFT unido

Se midieron los **18.588 registros TRAIN y 5.515 familias** del corpus congelado
`4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510`.
Se saltaron las 1.704 cabeceras VALIDATION antes de decodificar mensajes; no se
abrió TEST ni una suite reservada. No se cargaron pesos, Torch ni modelos, no se
ajustó el tokenizador y no se entrenó. Los archivos publicados contienen
agregados, hashes y código; no incluyen textos de conversaciones.

El BPE propio tiene 2.048 entradas y SHA-256
`848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720`.
La medición usó el helper nativo `dialogue_tokens`; reprodujo la máscara de
`_record_windows` sin importar el entrenador. **Las 18.588 ventanas enseñan
exactamente el último payload assistant más un EOS**, sin roles, usuarios ni
assistant anteriores como positivos. Todas caben en 512 entradas: cero recortes.

| Fuente | Registros | Familias | Tokens positivos medios, EOS incluido | Fracción de tokens positivos |
| --- | ---: | ---: | ---: | ---: |
| OASST1 humano revisado | 7.044 | 2.585 | 167,77 | 78,782 % |
| OASST2 humano revisado | 632 | 199 | 162,29 | 6,838 % |
| Composición procedimental del proyecto | 10.235 | 2.560 | 18,93 | 12,916 % |
| Diálogos nuevos escritos por IA del proyecto | 167 | 42 | 24,77 | 0,276 % |
| Lenguaje nuevo escrito por IA del proyecto | 153 | 39 | 27,29 | 0,278 % |
| Diálogos originales escritos por IA del proyecto | 357 | 90 | 38,21 | 0,909 % |

Los humanos representan **41,295 % de filas y 85,620 % de los 1.500.021 tokens
positivos**. Estos porcentajes describen el corpus: no miden la contribución
real de cada fuente al gradiente. La normalización por token favorece blancos
largos; la normalización por secuencia iguala el peso de las filas. La auditoría
no cambió la configuración encolada. La procedencia y licencias se conservan en
[DATASOURCES.md](../../../examples/learning-repair-20261013/DATASOURCES.md) y
[LICENSE-DATA.md](../../../examples/learning-repair-20261013/LICENSE-DATA.md).

| Longitud en BPE propio | Mediana | p90 | p95 | p99 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prompt antes de generar, marcador assistant incluido | 155 | 260 | 275 | 373 | 496 |
| Entradas teacher forcing, payload assistant incluido | 227 | 391 | 452 | 500 | 512 |
| Blanco positivo, EOS incluido | 33 | 258 | 332 | 404 | 444 |

La secuencia completa tiene una posición más que las entradas: máximo 513.
Todas las filas usan el mismo system por defecto, con 54 tokens de payload.
Los cuantiles usan rango entero `ceil(p*n)-1`. El contador de palabras usa
unidades Unicode de expresión regular, incluyendo números e identificadores
de código; no es un análisis lingüístico. OASST1 aporta 34.428 formas casefolded
distintas en blancos, frente a 74 en el banco de composición. Esos conteos no
demuestran comprensión ni cobertura semántica.

Hay **14.378 prompts completos proporcionados distintos y 12.002 blancos
exactos distintos**. Las 18.588 secuencias de registros son distintas, pero son
prefijos supervisados de 5.515 familias, no 18.588 conversaciones independientes.
Hay 2.649 prompts con varias respuestas. Las repeticiones dentro de cada fuente
se observan en OASST, que conserva ramas con alternativas; este conteo no las
declara contradicciones.
El reply más frecuente aparece 64 veces; 7.650 filas tienen un reply que también
aparece en otra fila. Igualdad de texto no equivale a igualdad de contexto.

La composición usa 40 plantillas completas, 64 familias por plantilla y 2.560
mundos declarados. Mantiene un vocabulario finito: 16 objetos, 26 nombres y 16
lugares. Cada plantilla aporta 254–256 registros tras retirar cinco duplicados.
Cuatro preguntas de último turno se repiten 640 veces cada una en estados
distintos; los prompts completos de composición siguen siendo 10.235 distintos.
Su diversidad de superficies y valores no acredita generalización conversacional.

La expansión repite mensajes previos como contexto: 36.213 apariciones assistant,
de las que 17.625 son historia sin supervisión positiva. Hay 18.601 mensajes
assistant distintos por familia/posición/contenido; los siete blancos erróneos
excluidos de TRAIN sobreviven solamente como historia correctiva. La octava
exclusión corresponde a VALIDATION y no fue auditada aquí. Las exclusiones se
documentan en el corpus; esta auditoría no los convirtió de nuevo en targets.

La medición tardó **11,068 s**, consumió 8,927 s de CPU y alcanzó 66,81 MiB RSS.
Se fijaron un CPU, paralelismo de tokenizadores desactivado, pared/CPU de 120 s,
espacio de direcciones de 1 GiB y archivo de salida de 32 MiB. La publicación
añadió únicamente conteos de IDs de familia y nombres precisos para métricas,
sin volver a tokenizar. Se conservan [receipt.json](receipt.json), la
[medición original](measurement-receipt-v1.json) y las dos fuentes ejecutadas.
El receipt publicado tiene SHA-256
`d1ff34f24b131b1c545a754c43110be3516632c72e9307437bf7fc69341f9452`.

La medición es reproducible desde la raíz del repositorio, sin entrenar:

```bash
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python docs/learning-repair-20261013/sft-data-audit/audit_sft_train.py \
  --repo . \
  --corpus examples/learning-repair-20261013/sft/sft.train-validation.jsonl \
  --tokenizer models/learning-repair-20261013/foundation-15m/intermediate-step004096/tokenizer.json \
  --output /ruta/nueva/audit-measurement.json

python docs/learning-repair-20261013/sft-data-audit/publish_sft_train_audit.py \
  --measurement /ruta/nueva/audit-measurement.json \
  --measurement-source docs/learning-repair-20261013/sft-data-audit/audit_sft_train.py \
  --corpus examples/learning-repair-20261013/sft/sft.train-validation.jsonl \
  --output /ruta/nueva/audit-publication
```

Las cuentas son deterministas con las fuentes fijadas; fechas, rutas y consumo
de recursos pueden variar. Los hashes del código nativo se registran antes y
después. La auditoría mide los datos y su supervisión, no entropía del modelo,
memorización efectiva, generación libre ni aceptación de conversación.
