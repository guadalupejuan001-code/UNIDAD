"""Post-selection review of both frozen fresh generations; no further fitting."""
from pathlib import Path
from review_common import write_review
root=Path(__file__).resolve().parent
# A clear false claim that a supplied fact was never supplied counts as critical.
# Garbled fragments are not completed into invented propositions by the reviewer.
write_review(root,'C7','fresh',[
 [('F',1,'Afirma que no se indicó la caja, aunque se especificó la redonda.'),('F',0,'Repetición sin describir el traslado.'),('F',0,'Fragmento sin identificar a Elvira.'),('F',0,'No distingue contenido de la caja y poseedora de la bolsa.')],
 [('F',0,'Responde sobre un material no solicitado, sin identificar el asiento.'),('F',0,'No identifica fila y asiento.'),('F',1,'Presenta doce como asiento del usuario después de corregirlo a catorce.'),('F',0,'No identifica la ubicación final.')],
 [('P',0,'Recupera Amalia, pero cambia el parentesco a mi prima.'),('F',0,'Fragmento sin separar posesión y propiedad.'),('F',1,'Afirma devolución ya realizada cuando sólo estaba prevista.'),('F',0,'No describe préstamo ni información temporal desconocida.')],
 [('P',0,'Recupera veinte minutos tarde, pero la oración está defectuosa.'),('F',0,'No recupera diez ni escribe el aviso actualizado.'),('P',0,'Recupera diez minutos tarde, pero omite ir en camino y añade fragmentos incoherentes.'),('F',0,'No describe la corrección de veinte a diez.')],
 [('F',1,'Niega haber recibido la ruta, aunque se indicó río con sombra.'),('F',0,'No identifica la ruta del puente.'),('F',1,'Indica la carretera descartada en vez de responder sobre la sombra desconocida.'),('F',1,'Indica la carretera descartada como ruta futura.')],
 [('F',0,'Habla de sombreros, sin responder quién lleva bufanda.'),('F',0,'Fragmento que no identifica a Ada.'),('F',0,'No identifica a Hugo como poseedor.'),('F',0,'No relaciona personas y prendas.')],
 [('F',0,'Negativa ininteligible sin distinguir entrega y recogida.'),('F',0,'No identifica el armario.'),('F',0,'Fragmento que no expresa que ahora lo tiene el usuario.'),('F',0,'Repetición sobre recepción sin reconocer que no se conoce remitente.')],
 [('F',1,'Niega haber recibido el martes, aunque se dio como propuesta.'),('C',0,'Recupera el miércoles; preposición menor defectuosa.'),('F',0,'Fragmento sin expresar claramente la hora desconocida.'),('F',1,'Niega conocer el día después de la confirmación del miércoles.')],
 [('F',1,'Afirma que no se indicó longitud, aunque se dieron nueve metros.'),('F',0,'No recupera la medida correcta.'),('F',0,'No responde si la cinta cambia la cuerda.'),('F',0,'Fragmento sin separar las tres medidas.')],
 [('F',0,'No identifica a Tomasa como organizadora.'),('P',0,'Recupera que Esteban tomó las fotos, pero no aclara completamente el papel del usuario y añade un fragmento.'),('F',0,'No identifica a Leandro después de la rectificación.'),('F',0,'No relaciona las tres personas con sus acciones.')]
])
write_review(root,'varied-s2-last','fresh',[
 [('F',1,'Niega que se haya dado información de los botones, ya especificada.'),('F',1,'Niega que se haya informado el traslado a la bolsa y repite fragmentos.'),('C',0,'Identifica a Elvira como poseedora.'),('F',0,'No identifica hilos y Elvira.')],
 [('F',0,'No recupera el asiento doce ni la fila ocho.'),('F',0,'Fragmento sin asiento ni fila correctos.'),('F',1,'Atribuye al usuario el asiento quince, que corresponde a Silvano.'),('F',0,'Menciona fila sin su número ni el asiento.')],
 [('F',1,'Niega haber recibido el dato de propiedad pese a que se indicó Amalia.'),('F',0,'Confunde mochila y préstamo sin separar posesión y propiedad.'),('P',0,'Niega que se haya devuelto; el resto no explica de forma inteligible cuándo ocurrirá.'),('F',0,'No explica préstamo ni hora desconocida.')],
 [('P',0,'Recupera llegaré veinte minutos, pero omite tarde y añade un comienzo defectuoso.'),('F',0,'No actualiza el aviso a diez minutos.'),('F',1,'Mantiene veinte minutos después de la corrección a diez.'),('F',0,'No describe la corrección de la demora.')],
 [('F',1,'Niega haber recibido el dato de sombra, que justificó la ruta inicial.'),('C',0,'Identifica la ruta del puente.'),('P',0,'Responde no, pero el añadido no explica claramente que falta el dato.'),('F',1,'Sitúa el puente como elección inicial y final, omitiendo y contradiciendo la inicial del río.')],
 [('F',0,'Habla de sombreros sin identificar quién lleva bufanda.'),('F',1,'Afirma que no se ha dicho quién lleva el sombrero, aunque Ada lo conserva.'),('F',0,'No identifica a Hugo.'),('F',0,'No relaciona correctamente personas y prendas.')],
 [('F',1,'Niega que se haya comunicado la llegada del paquete, dada explícitamente.'),('F',1,'Niega conocer quién lo guardó, aunque se indicó recepción; tampoco responde dónde.'),('F',0,'Fragmento sin expresar la recogida.'),('F',0,'Responde sobre precio desconocido en lugar del remitente.')],
 [('F',1,'Niega que se haya dado fecha, aunque el martes se indicó como propuesta.'),('F',0,'Responde sobre autor desconocido en vez del miércoles confirmado.'),('P',0,'Responde no, pero no expresa de forma clara que la hora sigue pendiente.'),('F',1,'Recupera martes como fecha después de confirmarse miércoles.')],
 [('F',1,'Niega haber recibido la medida de nueve metros, pese a que se acaba de dar.'),('C',0,'Recupera siete metros; preposición menor defectuosa.'),('F',1,'Mezcla siete y tres como medida de la cuerda, pese a ser objetos distintos.'),('F',0,'Fragmentos repetidos sin distinguir las tres medidas.')],
 [('F',0,'Repite exposición y fotos sin identificar a Tomasa.'),('P',0,'Recupera que sólo fue a verla, con cambio defectuoso de perspectiva a yo y sin identificar al fotógrafo.'),('F',1,'Relaciona a Leandro con tomar, la acción de las fotos, en vez de organizar.'),('F',0,'No relaciona a las personas con sus acciones.')]
])
