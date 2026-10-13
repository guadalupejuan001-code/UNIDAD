"""New situation families, frozen before this stage's training or selection."""
import hashlib,json,time
from pathlib import Path
root=Path(__file__).resolve().parent
cases=[
('container_contents',[
 ('En la caja redonda guardé botones. En la caja cuadrada guardé hilos. ¿Dónde están los botones?','En la caja redonda.',True),
 ('Moví los botones a una bolsa de tela. ¿Siguen en la caja redonda?','No; ahora están en la bolsa de tela.',True),
 ('La bolsa se la entregué a mi tía Elvira. ¿Quién tiene ahora los botones?','Tu tía Elvira.',True),
 ('¿Qué quedó en la caja cuadrada y quién tiene la bolsa?','En la caja cuadrada quedaron hilos; Elvira tiene la bolsa.',True)]),
('ticket_reassignment',[
 ('Mi entrada es para la fila ocho, asiento doce. ¿Qué asiento tengo?','El asiento doce de la fila ocho.',True),
 ('Cambian el asiento al catorce; la fila no cambia. ¿Dónde me sentaré?','En la fila ocho, asiento catorce.',True),
 ('A mi amigo Silvano le dieron el asiento quince. ¿Ese es mi asiento?','No; tu asiento es el catorce y el quince es el de Silvano.',True),
 ('Repite mi ubicación final sin incluir la de Silvano.','Fila ocho, asiento catorce.',True)]),
('loan_and_ownership',[
 ('La cámara es de mi prima Amalia. Me la prestó para hoy. ¿De quién es la cámara?','De tu prima Amalia.',True),
 ('Ahora la tengo yo en mi mochila. ¿Quién la tiene y quién es su dueña?','Tú la tienes; Amalia es su dueña.',True),
 ('La devolveré al terminar la visita. ¿Ya se la devolví?','No; dijiste que la devolverás al terminar la visita.',True),
 ('¿Qué sabemos del préstamo y qué no sabemos de cuándo termina?','La cámara de Amalia te fue prestada para hoy y la devolverás tras la visita; no diste una hora.',True)]),
('message_revision',[
 ('Quiero avisar: llegaré veinte minutos tarde. Escribe ese aviso en una frase.','Llegaré veinte minutos tarde.',False),
 ('Corrijo: serán diez minutos, no veinte. Actualiza el aviso.','Llegaré diez minutos tarde.',True),
 ('Añade que ya voy en camino. Mantén la demora corregida.','Ya voy en camino; llegaré diez minutos tarde.',True),
 ('¿Qué dato corregí en el aviso?','Cambiaste la demora de veinte a diez minutos.',True)]),
('choice_and_reason',[
 ('Elegí la ruta del río porque tiene sombra; descarté la carretera. ¿Por qué elegí esa ruta?','Porque la ruta del río tiene sombra.',False),
 ('La ruta del río quedó cerrada. Ahora iré por el puente. ¿Qué ruta usaré?','La del puente.',True),
 ('No te he dicho si en el puente hay sombra. ¿Puedes afirmarlo?','No; no has dado ese dato.',True),
 ('Distingue la razón de la elección inicial y la ruta que usaré finalmente.','Primero elegiste la ruta del río por la sombra; al cerrarla elegiste la del puente.',True)]),
('shared_and_distinct_properties',[
 ('Ada y Hugo llevan sombrero. Sólo Ada lleva bufanda. ¿Quién lleva bufanda?','Ada.',True),
 ('Hugo se quitó el sombrero. ¿Quién lo sigue llevando?','Ada.',True),
 ('Ada prestó su bufanda a Hugo. ¿Quién tiene ahora la bufanda?','Hugo.',True),
 ('¿Llevan ambos sombrero y quién tiene la bufanda ahora?','No: Ada lleva sombrero y Hugo tiene la bufanda.',True)]),
('delivery_state',[
 ('El paquete llegó al edificio, pero todavía no lo he recogido. ¿Ya lo tengo conmigo?','No; llegó al edificio, pero no lo has recogido.',True),
 ('La recepción lo guardó en el armario. ¿Dónde está?','En el armario de recepción.',True),
 ('Ya pasé por recepción y lo recogí. ¿Qué cambió?','Ahora tienes el paquete; antes estaba guardado en recepción.',True),
 ('¿Sabemos quién envió el paquete?','No has dicho quién lo envió.',True)]),
('provisional_and_confirmed',[
 ('La visita al taller está propuesta para el martes, pero aún no está confirmada. ¿Es una fecha segura?','No; es una propuesta todavía sin confirmar.',True),
 ('Confirmaron finalmente el miércoles. ¿Qué día quedó acordado?','El miércoles.',True),
 ('No nos dieron una hora. ¿Sabemos a qué hora empieza?','No; todavía no se ha dado una hora.',True),
 ('Resume qué está confirmado y qué sigue pendiente.','La visita al taller está confirmada para el miércoles; falta la hora.',True)]),
('measurement_correction',[
 ('Anoté que la cuerda mide nueve metros. ¿Qué longitud anoté?','Nueve metros.',False),
 ('La medí de nuevo: mide siete metros. La primera medida era incorrecta. ¿Cuál es la correcta?','Siete metros.',True),
 ('La cinta, que es otro objeto, mide tres metros. ¿Eso cambia la longitud de la cuerda?','No; la cuerda sigue midiendo siete metros.',True),
 ('Distingue la medida inicial de la cuerda, su medida corregida y la de la cinta.','La cuerda se anotó como nueve metros y se corrigió a siete; la cinta mide tres.',True)]),
('roles_in_event',[
 ('Tomasa organizó la exposición y Esteban tomó las fotos. ¿Quién la organizó?','Tomasa.',True),
 ('Yo sólo fui a verla. ¿Fui quien tomó las fotos?','No; las tomó Esteban y tú fuiste a ver la exposición.',True),
 ('Rectifico: quien organizó fue Leandro. Tomasa presentó las obras. ¿Quién organizó al final?','Leandro.',True),
 ('Relaciona a Leandro, Tomasa y Esteban con lo que hizo cada uno.','Leandro organizó, Tomasa presentó las obras y Esteban tomó las fotos.',True)])]
suite={'schemaVersion':1,'purpose':'Fresh transfer across new situation families; not consulted until candidate selection is frozen.',
       'frozenAtUnix':time.time(),'generation':{'temperature':0,'seed':42,'maxNewTokens':32},
       'familySeparation':'Situation structures absent from this stage corpus; language and basic retrieval/correction operations can be shared.',
       'review':'Project AI semantic review, not human or blinded. Expected wording is illustrative, not exact-match grading.',
       'episodes':[]}
for i,(family,turns) in enumerate(cases,1):
 e={'id':f'principles-transfer-{i:02}','family':family,'turns':[]}
 for j,(user,expected,critical) in enumerate(turns,1):
  e['turns'].append({'id':f'{e["id"]}-t{j}','user':user,'expectedMeaning':expected,
                   'critical':critical,'requirements':['Preserve the requested facts and latest corrections.','Respond intelligibly without inventing facts or claiming unperformed actions.']})
 suite['episodes'].append(e)
p=root/'fresh-transfer.json';p.write_text(json.dumps(suite,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'turns':40}))
