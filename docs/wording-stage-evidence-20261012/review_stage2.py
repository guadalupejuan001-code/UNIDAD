"""Semantic review by project AI, after generation; not an independent panel."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
reviews={
'varied-s2-best':[
 [('F',1,'Convierte el apodo en destino.'),('P',0,'Recupera faro, pero no la acción de dibujarlo.'),('F',1,'Convierte el nuevo apodo en destino.'),('P',0,'Recuerda el apodo actual, pero omite el primero.')],
 [('C',0,'Reconoce que desconoce el color.'),('F',0,'Repite color sin recuperar coral.'),('P',0,'Recupera Nelavia, pero cambia la perspectiva del parentesco a mi prima.'),('F',0,'No recupera el color y responde sobre precio en vez de tienda.')],
 [('C',0,'Identifica la ubicación actual, con añadido defectuoso de persona en el viaje.'),('F',0,'Repetición larga que agota la salida.'),('P',0,'Niega la llegada; el fragmento posterior no conserva claramente la ubicación.'),('P',0,'Conserva el origen y omite el destino.')],
 [('C',0,'Escucha sin aconsejar.'),('C',0,'Pregunta quién interrumpió.'),('P',0,'Recuerda la interrupción sin identificar al compañero.'),('C',0,'Resume quién interrumpió y qué sucedió.')],
 [('P',0,'Recupera cuentos y parte del nombre deformado.'),('F',0,'Mezcla experiencias, apodo y abuelo sin origen inteligible.'),('F',0,'Fragmento sin explicar cambio y continuidad.'),('F',0,'Responde con el cambio de nombre, sin relación con el abuelo.')],
 [('F',1,'Atribuye a Darielvo el parentesco de hermana.'),('F',1,'Identifica a Dariel como hermana y no a Selvaria.'),('F',0,'No identifica de forma inteligible a quien cuida los cactus.'),('F',1,'Introduce Selvo como hermana y omite las plantas.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne día, hora y lugar actualizados.')],
 [('P',0,'Recupera plantas como primera actividad sin expresar regarlas.'),('C',0,'Indica revisar el borrador primero, con flexión defectuosa menor.'),('F',1,'Coloca revisar después del propio borrador.'),('F',1,'Presenta regar un borrador como orden anterior y actual.')],
 [('C',0,'Recupera el título.'),('C',0,'Reconoce que no se ha dado autor; el añadido de Sal es defectuoso menor.'),('P',0,'Recupera finalidad y día, pero los expresa desde mi en lugar de tu.'),('F',0,'Fragmento sin separar información conocida y desconocida.')],
 [('F',0,'Habla de ayudar al perro y omite pasear.'),('F',0,'No identifica pasear antes de escribir.'),('F',0,'No recupera el nuevo momento.'),('F',0,'Repetición sin relacionar las actividades.')]],
'varied-s2-last':[
 [('F',1,'Convierte el apodo en destino.'),('P',0,'Recupera faro sin la acción de dibujarlo.'),('P',0,'Apodo actual incompleto: Zela.'),('F',0,'Fragmento que no distingue apodos anterior y actual.')],
 [('C',0,'Reconoce que desconoce el color.'),('F',0,'Repite color sin recuperar coral.'),('P',0,'Recupera parte de Nelavia; perspectiva mi prima defectuosa.'),('F',0,'No recupera color y responde sobre precio en vez de tienda.')],
 [('F',1,'Presenta Puerto Arce como ubicación de hoy.'),('P',0,'Recupera parcialmente el nuevo destino con repetición.'),('C',0,'Niega la llegada y conserva Valdehumo.'),('P',0,'Recupera el origen y un fragmento del destino sin relación clara.')],
 [('C',0,'Escucha sin aconsejar.'),('C',0,'Pregunta quién interrumpió.'),('C',0,'Identifica al compañero Taviro.'),('C',0,'Resume quién interrumpió y qué sucedió.')],
 [('P',0,'Se reconocen cuentos y parte del nombre deformado.'),('P',0,'Se reconoce al abuelo como origen, pero no sus historias ni perspectiva clara.'),('P',0,'Recupera el nuevo título y omite lo conservado.'),('F',0,'Mezcla nombre, historias y apodo sin explicar la relación.')],
 [('F',1,'Asigna a Darielvo el parentesco de hermana.'),('F',1,'Asigna el violín a Dariel.'),('P',0,'Se reconoce parte de Selvaria como cuidadora, con nombre incompleto.'),('F',1,'Introduce Selvo como hermana y agota salida por repetición.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne día, hora y lugar actualizados.')],
 [('P',0,'Recupera plantas como primera actividad sin expresar regarlas.'),('C',0,'Indica revisar el borrador primero, con flexión defectuosa menor.'),('F',1,'Coloca revisar el borrador después del propio borrador.'),('P',0,'Recupera la prioridad actual; deja ambiguo el orden anterior.')],
 [('C',0,'Recupera el título completo.'),('C',0,'Reconoce que no se ha dado el autor.'),('P',0,'Recupera finalidad y día con cambio defectuoso de perspectiva a mi.'),('F',0,'No identifica el libro ni separa de forma inteligible lo conocido de lo desconocido.')],
 [('F',0,'Habla de ayudar al perro y omite pasear.'),('F',0,'No identifica pasear antes de escribir.'),('F',0,'Fragmento que no recupera anochecer.'),('F',0,'No relaciona paseo y escritura.')]]}

for tag,episodes in reviews.items():
 p=root/f'{tag}.known.actual.json';d=json.loads(p.read_text());rows=[]
 assert len(episodes)==len(d['episodes'])==10
 for episode,annotations in zip(d['episodes'],episodes):
  assert len(episode['turns'])==len(annotations)==4
  for turn,(grade,critical,reason) in zip(episode['turns'],annotations):
   rows.append({'id':turn['id'],'grade':grade,'critical':bool(critical),'reason':reason})
 counts={g:sum(r['grade']==g for r in rows) for g in ['C','P','F']}
 summary={**counts,'criticalFailureTurns':sum(r['critical'] for r in rows),'completeDialogs':sum(all(rows[i+j]['grade']=='C' for j in range(4)) for i in range(0,40,4)),'requestedTurns':40,'reviewedTurns':40,'reductionInFVsC7':(28-counts['F'])/28}
 out={'modelId':d['modelId'],'actualSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'suiteSha256':d['suiteSha256'],'generation':d['generation'],'rubricFile':'unidad-fix/grading-rubric.json','reviewer':{'kind':'project AI','human':False,'independent':False,'blinded':False},'turns':rows,'summary':summary,'goal50PercentReached':counts['F']<=14 and summary['criticalFailureTurns']<=6,'productionAccepted':False,'freshTransferExamined':False}
 (root/f'{tag}.known.review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(tag,summary)
