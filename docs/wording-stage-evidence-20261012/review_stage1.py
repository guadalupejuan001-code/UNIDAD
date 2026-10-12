"""Manual project-AI judgments of saved outputs; never part of generation."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent

reviews={
'varied-s1-best':[
 [('F',1,'Usa el apodo como destino.'),('C',0,'Conserva dibujar un faro, con redundancia menor.'),('F',1,'Usa dibujar como apodo.'),('F',1,'Presenta Zela como la preferencia anterior, en vez de Zelún.')],
 [('F',0,'Negativa ininteligible sin identificar el color desconocido.'),('F',0,'Dice color, pero no recupera coral.'),('P',0,'Recupera Nelavia; la perspectiva mi prima no conserva claramente la del usuario.'),('F',0,'Habla del regalo en vez de separar color conocido y tienda desconocida.')],
 [('P',0,'Recupera Valde como fragmento de la ubicación.'),('F',0,'No responde al nuevo destino.'),('C',0,'Niega la llegada y conserva Valdehumo.'),('F',1,'Añade Valce como ubicación en el resumen.')],
 [('C',0,'Invita a contar lo ocurrido sin aconsejar.'),('C',0,'Pregunta quién interrumpió; flexión verbal defectuosa pero intención inteligible.'),('C',0,'Identifica al compañero Taviro.'),('P',0,'Recupera a Taviro, pero deja sin expresar la situación.')],
 [('P',0,'Recupera cuentos y parte del título, deformado.'),('P',0,'Se reconocen historias y abuelo, con sintaxis y perspectiva defectuosas.'),('F',0,'No distingue el cambio de título y el contenido conservado.'),('F',0,'Fragmento sin conexión inteligible con el abuelo.')],
 [('F',0,'No identifica personas ni relaciones de forma inteligible.'),('F',1,'Asigna a Darielvo el parentesco de hermana.'),('F',0,'Repite la pregunta sin identificar quién cuida los cactus.'),('F',1,'Introduce Selvo como hermana y no responde la segunda parte.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne los tres datos actualizados.')],
 [('F',1,'Introduce regar un borrador como primera actividad.'),('F',1,'Mantiene regar un borrador como prioridad.'),('F',1,'Pone revisar el borrador después del propio borrador.'),('F',1,'Describe regar un borrador como orden previo y actual.')],
 [('F',0,'Título deformado e ininteligible.'),('C',0,'Reconoce que no se ha dicho quién lo escribió; Sal queda como añadido defectuoso menor.'),('F',0,'No expresa la finalidad del libro.'),('F',0,'Fragmento sin separar información conocida y desconocida.')],
 [('F',0,'No identifica pasear como ayuda.'),('F',0,'No expresa qué actividad precede a escribir.'),('F',0,'No recupera el nuevo momento del paseo.'),('F',0,'No relaciona las actividades.')]],
'varied-s1-last':[
 [('F',1,'Usa el apodo como destino.'),('P',0,'Recupera faro, pero omite la intención de dibujarlo.'),('P',0,'Recupera parcialmente el apodo vigente como Zela; queda incompleto.'),('P',0,'Sólo recupera parcialmente el apodo actual; omite el anterior.')],
 [('C',0,'Reconoce que no se ha dado el color.'),('F',0,'Repite color sin recuperar coral.'),('F',0,'No identifica a quien dio el paraguas.'),('F',0,'No distingue color conocido y tienda desconocida.')],
 [('C',0,'Identifica Valdehumo como ubicación; el añadido del viaje conserva los datos con cambio defectuoso de persona.'),('F',0,'Repetición larga y salida agotada; no constituye una respuesta inteligible completa.'),('C',0,'Niega la llegada y conserva la ubicación actual.'),('P',0,'Conserva el origen y omite el destino.')],
 [('C',0,'Invita a contar lo ocurrido sin aconsejar.'),('C',0,'Pregunta por quien interrumpió, con flexión verbal menor.'),('P',0,'Recuerda la interrupción, pero no identifica al compañero.'),('P',0,'Recupera Taviro sin resumir la situación.')],
 [('P',0,'Se reconocen proyecto, parte del título y cuentos; hay deformaciones.'),('F',0,'Fragmento ininteligible sobre el origen.'),('F',0,'No identifica el cambio ni lo conservado.'),('F',1,'Afirma que el abuelo es Cuaderno de Junco.')],
 [('F',1,'Identifica a Darielvo como hermana.'),('F',0,'Devuelve una pregunta sin responder.'),('F',0,'Nombre y predicado deformados, sin identificación inteligible.'),('F',1,'Asigna cactus a Darielvo después de la corrección a helechos y deforma el otro referente.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne día, hora y lugar actualizados.')],
 [('P',0,'Menciona ambas actividades, empezando por plantas, sin expresar claramente la primera acción.'),('C',0,'Indica revisar el borrador primero, con flexión defectuosa menor.'),('F',1,'Pone el borrador después del propio borrador.'),('F',0,'Fragmento sin describir el cambio de orden.')],
 [('P',0,'Recupera el título completo, pero lo precede una repetición que deteriora la respuesta.'),('P',0,'Reconoce la falta de autor con repetición larga y un añadido defectuoso.'),('F',0,'Mezcla fragmentos de lectura y apodo sin finalidad inteligible.'),('F',0,'No separa el título, la finalidad y el autor desconocido.')],
 [('F',0,'No identifica la actividad que despeja.'),('F',0,'No identifica qué ocurre antes de escribir.'),('F',0,'No recupera anochecer.'),('F',0,'No relaciona paseo y escritura.')]]}

for tag,episodes in reviews.items():
 p=root/f'{tag}.known.actual.json';d=json.loads(p.read_text());rows=[]
 assert len(episodes)==len(d['episodes'])==10
 for episode,annotations in zip(d['episodes'],episodes):
  assert len(episode['turns'])==len(annotations)==4
  for turn,(grade,critical,reason) in zip(episode['turns'],annotations):
   rows.append({'id':turn['id'],'grade':grade,'critical':bool(critical),'reason':reason})
 counts={g:sum(r['grade']==g for r in rows) for g in ['C','P','F']}
 summary={**counts,'criticalFailureTurns':sum(r['critical'] for r in rows),
          'completeDialogs':sum(all(rows[i+j]['grade']=='C' for j in range(4)) for i in range(0,40,4)),
          'requestedTurns':40,'reviewedTurns':40,'reductionInFVsC7':(28-counts['F'])/28}
 out={'modelId':d['modelId'],'actualSha256':hashlib.sha256(p.read_bytes()).hexdigest(),
      'suiteSha256':d['suiteSha256'],'generation':d['generation'],
      'rubricFile':'unidad-fix/grading-rubric.json','reviewer':{'kind':'project AI','human':False,'independent':False,'blinded':False},
      'turns':rows,'summary':summary,'goal50PercentReached':counts['F']<=14 and summary['criticalFailureTurns']<=6,
      'productionAccepted':False,'freshTransferExamined':False}
 (root/f'{tag}.known.review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(tag,summary)
