"""Assemble measured results after candidate freeze and fresh semantic review."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
choice=json.loads((root/'candidate-selection.json').read_text())
known=json.loads((root/f"{choice['label']}.known.review.json").read_text())
fresh=json.loads((root/f"{choice['label']}.fresh.review.json").read_text())
baseline=json.loads((root/'C7.fresh.review.json').read_text())
status=json.loads((root/'varied-run/status.json').read_text())
manifest=json.loads((root/choice['label']/'manifest.json').read_text())
cli=json.loads((root/'candidate-cli-verification.json').read_text())
all_reviews=[]
for path in sorted(root.glob('varied-s*.known.review.json')):
 review=json.loads(path.read_text());tag=path.name.removesuffix('.known.review.json')
 generation=json.loads((root/f'{tag}.generation-summary.json').read_text())
 model=json.loads((root/tag/'manifest.json').read_text())
 all_reviews.append({'label':tag,'step':model['training']['selectedStep'],'derivation':model['training'].get('derivation'),'modelId':review['modelId'],**review['summary'],
                     'wordingExact':generation['wordingExact'],'validationExact':generation['validationExact']})
all_reviews.sort(key=lambda x:x['step'])
b=baseline['summary'];f=fresh['summary'];k=known['summary']
fresh_reduction=(b['F']-f['F'])/b['F'] if b['F'] else None
results={'selectedLabel':choice['label'],'modelId':choice['modelId'],'knownReference':{'C':7,'P':5,'F':28,'criticalFailureTurns':6,'completeDialogs':0},
         'knownCandidate':k,'knownFReduction':(28-k['F'])/28,'knownGoal50PercentReached':known['goal50PercentReached'],
         'freshReference':b,'freshCandidate':f,'freshFReduction':fresh_reduction,
         'freshGoal50PercentReached':bool(b['F'] and f['F']<=b['F']/2 and f['criticalFailureTurns']<=b['criticalFailureTurns']),
         'productionAccepted':False,'reviewer':'one project AI, not human, independent or blinded',
         'allDevelopmentCandidates':all_reviews,'finalTrainingStatus':status,'candidateFiles':manifest['files'],'cliVerification':cli,
         'hormigaImportCompatible':False,'hormigaRuntimeImplemented':False}
(root/'final-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
rows=['| C7, referencia | — | 7 | 5 | 28 | 6 | 0/10 | — | — |']
for r in all_reviews:
 step_text='media 448+679' if r['derivation'] else str(r['step'])
 rows.append(f"| {r['label']} | {step_text} | {r['C']} | {r['P']} | {r['F']} | {r['criticalFailureTurns']} | {r['completeDialogs']}/10 | {r['wordingExact']}/8 | {r['validationExact']}/40 |")
report=f'''# Unidad: generalización de redacción y prueba propia

12 de octubre de 2026. Candidato experimental: **{choice['label']}**, paso
{manifest['training']['selectedStep']} de esta etapa. No se promociona como
asistente competente ni cambia el modelo predeterminado.

En los 40 turnos conocidos, fallidas pasó de 28 a {k['F']}
({(28-k['F'])/28:.1%} de reducción); completas pasó de 7 a {k['C']},
críticos de 6 a {k['criticalFailureTurns']} y diálogos completos de 0 a
{k['completeDialogs']}/10. Son casos de desarrollo usados para seleccionar.
En 40 turnos nuevos, C7 obtuvo C{b['C']}/P{b['P']}/F{b['F']} y el candidato
C{f['C']}/P{f['P']}/F{f['F']}; los críticos fueron
{b['criticalFailureTurns']} y {f['criticalFailureTurns']}, respectivamente.
La reducción nueva fue {fresh_reduction:.1%}. No confundir ambos resultados.

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
BF16 en entrenamiento, inferencia FP32. Fin observado: paso {status['step']},
{status['elapsedSeconds']:.3f} segundos, motivo nativo `{status['reason']}`.
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
{chr(10).join(rows)}

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
| C7 | {b['C']} | {b['P']} | {b['F']} | {b['criticalFailureTurns']} | {b['completeDialogs']}/10 |
| {choice['label']} | {f['C']} | {f['P']} | {f['F']} | {f['criticalFailureTurns']} | {f['completeDialogs']}/10 |

La aceptación del proyecto sigue exigiendo 36 C/40, ocho diálogos completos
y ningún fallo crítico o técnico. No se ha alcanzado. La meta de reducir F un
50% en desarrollo figura como `{results['knownGoal50PercentReached']}` y en
esta comparación nueva como `{results['freshGoal50PercentReached']}`; no son
una estimación universal de calidad.

## Pesos e instrucción de carga

Modelo exacto: `{choice['modelId']}`.

- Pesos SHA-256: `{manifest['files']['weights.safetensors']['sha256']}`.
- Tokenizador SHA-256: `{manifest['files']['tokenizer.json']['sha256']}`.
- Corpus nuevo SHA-256: `add631f57bd07f4a5cac85ed30a226670a2701b4a41781b4a0db3a49859f3449`.
- Prueba nueva SHA-256: `9af622085a17e2915f3d967806203128be4632e2cdda55bab19731716f296701`.

El paquete `Unidad-Prueba-Propia.zip` conserva todos los pesos evaluados,
referencia, punto de partida, estado final nativo con optimizador y corpus.
Las generaciones, revisiones y scripts se conservan en este repositorio.
Descomprimir y ejecutar `python restaurar.py
DIRECTORIO_NUEVO`; el restaurador verifica hashes y no sobrescribe directorios.
Con Unidad y sus dependencias conversacionales instaladas, desde ese directorio:

```bash
python -m ia_propia conversation converse --model models/{choice['label']} \\
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
'''
(root.parent/'UNIDAD-principios/docs/unidad-wording-stage-20261012.md').write_text(report)
print(json.dumps(results,ensure_ascii=False),flush=True)
