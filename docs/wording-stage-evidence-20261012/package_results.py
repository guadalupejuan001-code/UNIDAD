"""Preserve all evaluated weights and final native state, deduplicated by bytes."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
choice=json.loads((root/'candidate-selection.json').read_text())
results=json.loads((root/'final-results.json').read_text())
status=json.loads((root/'varied-run/status.json').read_text())
assert status['status']=='stopped', 'Do not package a changing training run'
output=root.parent/'Unidad-Prueba-Propia.zip'
members={};sources={}
def add(source,relative):
 data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
 members[relative]={'sha256':sha,'bytes':len(data),'object':'objects/'+sha}
 sources.setdefault(sha,source)
models=sorted(p for p in root.glob('varied-s*') if p.is_dir() and (p/'manifest.json').exists())
models += [root.parent/'unidad-fix/pointer-v1-best',root.parent/'UNIDAD/models/conversation-reference-c7']
for model in models:
 for name in ['config.json','tokenizer.json','weights.safetensors','manifest.json']:
  add(model/name,f'models/{model.name}/{name}')
run=root/'varied-run'
for checkpoint in sorted({status['lastCheckpoint'],status['bestCheckpoint']}):
 for path in sorted((run/'checkpoints'/checkpoint).iterdir()):
  if path.is_file():add(path,f'training/varied-run/checkpoints/{checkpoint}/{path.name}')
for name in ['status.json','events.jsonl']:
 add(run/name,f'training/varied-run/{name}')
add(root/'varied-corpus.jsonl','corpus/varied-corpus.jsonl')
add(root.parent/'unidad-fix/repair-corpus.jsonl','corpus/previous-repair-corpus.jsonl')
index={'schemaVersion':1,'selectedModel':choice['label'],'selectedModelId':choice['modelId'],
       'productionAccepted':False,'finalResults':results,'members':members}
with zipfile.ZipFile(root.parent/'Unidad-Pesos-Diagnostico.zip') as old:
 restore=old.read('restaurar.py')
readme=f'''UNIDAD: PESOS PROPIOS Y PRUEBA REPRODUCIBLE — 12 OCTUBRE 2026

Descomprime este ZIP y ejecuta:
    python restaurar.py DIRECTORIO_NUEVO
Verifica hashes antes de escribir y no sobrescribe un destino existente.

Candidato experimental seleccionado: {choice['label']}
Identidad exacta: {choice['modelId']}
Resultados y límites, código y evaluaciones: PR 2 del repositorio privado UNIDAD.
https://github.com/guadalupejuan001-code/UNIDAD/pull/2
Informe: docs/unidad-wording-stage-20261012.md
No se ha aceptado para producción ni como asistente conversacional competente.

Con el código de la PR 2 de UNIDAD y sus dependencias conversacionales instalados,
entra en el directorio restaurado y ejecuta:
    python -m ia_propia conversation converse --model models/{choice['label']} --temperature 0 --seed 42 --max-new-tokens 32

Para reproducir las generaciones, usa el evaluador y los casos incluidos en
docs/wording-stage-evidence-20261012/ de esa rama, indicando la ruta de estos
pesos restaurados. La prueba nueva publicada ya no es ciega para trabajo futuro.

models/ conserva todos los artefactos evaluados en esta etapa, la referencia C7
y el punto de partida propio. training/ conserva el estado nativo final y el
mejor según pérdida, con optimizador. El presupuesto terminó; no hay un proceso
en ejecución ni restaurar el paquete lo inicia. Las rutas absolutas del estado
son las de la ejecución original. Toda etapa posterior requiere configuración
y presupuesto explícitos; no reiniciar un presupuesto ya cerrado.

El repositorio conserva los scripts históricos, generaciones, revisiones y el
beam descartado. Éste no se usa al probar estos modelos. corpus/ contiene sólo
datos ficticios de entrenamiento.
No se usaron conversaciones personales ni pesos de otra IA.

Hormiga X: este paquete no es GGUF y su importador actual no lo carga. Consulta
docs/unidad-hormiga-compatibility-20261012.md en GitHub. No se creó ni probó un APK.
'''
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
 archive.writestr('LEEME.txt',readme)
 archive.writestr('restaurar.py',restore)
 archive.writestr('indice.json',json.dumps(index,ensure_ascii=False,indent=2)+'\n')
 for sha,source in sources.items():archive.write(source,'objects/'+sha)
with zipfile.ZipFile(output) as archive:
 bad=archive.testzip()
 if bad:raise ValueError('ZIP CRC failed: '+bad)
receipt={'path':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
         'models':len(models),'restoredFiles':len(members),'distinctPayloadObjects':len(sources),'zipCrcPassed':True}
(root/'weights-package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
