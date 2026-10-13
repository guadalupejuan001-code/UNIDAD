# Candidato propio de copia: 128 pasos, rechazado

Este artefacto experimental parte exactamente de C7. El entrenamiento cambia
los cuatro tensores de copia (16.641 parámetros) y conserva los otros 6.967.296
byte a byte. El delta contiene todos los cambios de pesos; C7 contiene la base.

**No superó la evaluación nueva:** 0/40 completas, 0/10 diálogos completos y ocho
críticas, frente a cuatro críticas de C7. Dos revisiones de IA del proyecto
coinciden en esos recuentos. Se conserva para reproducir el experimento; no se
recomienda como una mejora de conversación ni se promueve como modelo.

En 16 turnos de desarrollo conocidos corrige la pertenencia de una linterna
después de corregir sólo otro objeto: responde «La linterna es de Raúl.» donde
C7 respondía «Queda de Elisa.». No demuestra conversación general ni sustituye
una IA comercial. C7 se conserva como referencia y este modelo no se promueve.

Para reconstruir y cargar exactamente el candidato evaluado desde este clon:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[conversation]'
PYTHONPATH=src python models/conversation-copy-head-20261013/reconstruir.py
PYTHONPATH=src python -m ia_propia.conversation converse \
  --model models/conversation-copy-head-20261013 \
  --max-new-tokens 160 --temperature 0 --seed 42
```

La reconstrucción verifica la base, el delta y el archivo final de pesos antes
de cargar el manifiesto. Configuración y tokenizador mantienen sus identidades;
el manifiesto identifica un modelo nuevo de linaje propio. No carga pickle.
Los resultados completos y las condiciones se conservan en el
[informe del experimento](../../docs/work-brain-nerva-20261013.md).
