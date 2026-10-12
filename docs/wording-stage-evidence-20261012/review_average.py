from pathlib import Path
from review_common import write_review
root=Path(__file__).resolve().parent
write_review(root,'varied-s3-mean','known',[
 [('F',1,'Convierte el apodo en destino.'),('C',0,'Recuerda dibujar un faro.'),('C',0,'Recupera el nuevo apodo completo.'),('P',0,'Recupera el apodo actual, pero omite el anterior y repite ahora.')],
 [('F',0,'Negativa ininteligible sin identificar el color desconocido.'),('F',0,'Dice color sin recuperar coral.'),('P',0,'Recupera parte de Nelavia y cambia la perspectiva a mi prima.'),('F',0,'No recupera el color y responde sobre precio en vez de tienda.')],
 [('F',1,'Presenta Puerto Arce como ubicación de hoy.'),('P',0,'Recupera parcialmente el nuevo destino, con repetición.'),('C',0,'Niega la llegada y conserva Valdehumo.'),('F',0,'Repetición larga que agota la salida, sin destino ni resumen inteligible.')],
 [('C',0,'Escucha sin aconsejar.'),('C',0,'Pregunta quién interrumpió.'),('P',0,'Recuerda la interrupción, pero no quién fue.'),('C',0,'Resume quién interrumpió y qué sucedió.')],
 [('P',0,'Recupera proyecto y parte del nombre, sin contenido.'),('P',0,'Se reconoce al abuelo como origen, sin historias ni perspectiva claras.'),('P',0,'Recupera el nuevo título y omite lo conservado.'),('P',0,'Relaciona proyecto, experiencias y abuelo sin claridad suficiente.')],
 [('F',1,'Atribuye a Darielvo el parentesco de hermana.'),('P',0,'Recupera hermana como quien toca, pero usa mi hermana y omite su nombre.'),('P',0,'Recupera parcialmente Selvaria como Sel.'),('F',1,'Introduce Selvo como hermana y agota salida por repetición.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne día, hora y lugar actualizados.')],
 [('P',0,'Recupera plantas como primera actividad sin expresar regarlas.'),('C',0,'Indica revisar el borrador primero, con flexión defectuosa menor.'),('F',1,'Introduce regar un borrador como actividad posterior.'),('F',1,'Presenta regar un borrador como actividad previa.')],
 [('C',0,'Recupera el título completo.'),('C',0,'Reconoce que desconoce quién lo escribió.'),('P',0,'Recupera finalidad y día con cambio de perspectiva a mi.'),('F',0,'Fragmento sin identificar información conocida y desconocida.')],
 [('F',0,'Fragmento sin recuperar pasear como ayuda.'),('F',0,'No identifica pasear antes de escribir.'),('F',0,'No recupera anochecer de forma inteligible.'),('F',0,'Habla de preferencias sin relacionar paseo y escritura.')]
],28)
