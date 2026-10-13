# DEV40 después de Tatoeba y retención 1.024

El candidato propio `60d94233…`, BEST de retención 1.024 después de BASE Tatoeba 2.048, desciende de `3ccd6deb… → 22c1debc… → 60d94233…`, con origen en pesos aleatorios propios. Ambas fases conservan motivo nativo `signal`, TRAIN/VALIDATION separados y TEST ausente/no evaluado.

Obtuvo **0 completas, 0 parciales, 40 fallidas y 0/10 diálogos** en ambos juicios de IA del proyecto. Primaria: **9 críticas**; cruzada: **10 críticas**. Ambas: **5 técnicas por token_budget**, 35 EOS, sin excepciones ni descartes de contexto. No hay aceptación.

La crítica adicional cruzada es `cinema_age_correction`. [Summary](comparison-summary.json) conserva todas las discrepancias por dimensión y los turnos que la cruzada marca como interpretación incierta. Los juicios originales no se corrigen ni se convierten en consenso humano.

[Integridad](publication-integrity.json) registra 52 copias byteexact: 40 turnos, outputs/template, ambos juicios y assessments, los tres ficheros congelados de suite y tres JSON de metadatos del artefacto. [Proof de historiales](raw-history-proof.json) confirma usuarios y respuestas previas reales, identidad, generación y ambos assessments con el verificador original, sin cargar modelos. Cada campo del runner coincide; los campos adicionales de la primaria también se conservan intactos.

Las respuestas y metadatos están disponibles en este paquete local publicable. Los pesos completos y el checkpoint nativo siguen **privados en el momento de este archivo**; no se publican aquí ni se presenta source-artifact como modelo ejecutable completo. Corpus y linaje constan en [manifest](source-artifact/manifest.json); su encabezado no equivale a exposición completa de TRAIN ni conocimiento aprendido.

El [endpoint español LATAM v3 anterior](../development-wide-latam-retention1024-v1/README.md) conserva 1/40 completa, seis críticas/dos técnicas. Aquí se observa deterioro en completas y fallos bajo la misma rúbrica, contexto 2.048 y generación 160 tokens/temperatura 0/semilla 42. Cambian dos fases, pesos y datos: no se atribuye el resultado a una sola variable ni se comparan NLL de corpus distintos. Ambos fallan 36/40, 8/10 y cero críticas/técnicas. Esta prueba es DEV expuesta; no se abrió test reservado.
