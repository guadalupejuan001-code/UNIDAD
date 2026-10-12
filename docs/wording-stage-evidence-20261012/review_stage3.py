from pathlib import Path
from review_common import write_review
root=Path(__file__).resolve().parent
write_review(root,'varied-s3-last','known',[
 [('F',1,'Convierte el apodo en destino.'),('C',0,'Recuerda dibujar un faro.'),('C',0,'Recupera el nuevo apodo completo.'),('P',0,'Recuerda el apodo actual y omite el primero, con repetición y un fragmento final.')],
 [('C',0,'Reconoce que desconoce el color.'),('F',0,'Repite color sin recuperar coral.'),('P',0,'Recupera Nelavia con cambio de perspectiva a mi prima.'),('F',0,'No recupera color y responde sobre precio en vez de tienda.')],
 [('C',0,'Identifica Valdehumo; el añadido del viaje cambia defectuosamente de persona.'),('P',0,'Recupera Villa Nébula entre repeticiones y fragmentos, sin agotar salida.'),('P',0,'Niega la llegada y recupera Valdehumo, con cambio de perspectiva a sigo.'),('P',0,'Conserva el origen y omite el destino.')],
 [('C',0,'Escucha sin aconsejar.'),('C',0,'Pregunta quién interrumpió, con flexión defectuosa menor.'),('C',0,'Identifica al compañero Taviro, con flexión y repetición menores.'),('C',0,'Resume quién interrumpió y qué sucedió.')],
 [('P',0,'Recupera cuentos y parte del título, deformado.'),('P',0,'Se reconoce al abuelo como origen, sin recuperar claramente sus historias.'),('F',0,'Fragmento sin identificar el cambio y lo conservado.'),('P',0,'Relaciona el proyecto con experiencias y abuelo, pero sin perspectiva ni relación claras.')],
 [('F',0,'No identifica a las dos personas.'),('P',0,'Recupera hermana como quien toca, pero usa mi hermana y omite su nombre.'),('P',0,'Recupera parcialmente Selvaria como Sel.'),('F',1,'Introduce Selvo como hermana y agota salida por repetición.')],
 [('C',0,'Jueves a las cinco.'),('C',0,'Sábado a las cinco.'),('C',0,'Biblioteca del Olmo.'),('C',0,'Reúne día, hora y lugar actualizados.')],
 [('F',1,'Introduce regar un borrador como actividad.'),('F',1,'Sustituye revisar por regar un borrador.'),('F',1,'Afirma regar un borrador como actividad posterior.'),('F',1,'Presenta regar un borrador como orden previo.')],
 [('C',0,'Recupera el título completo.'),('C',0,'Reconoce que desconoce quién lo escribió.'),('P',0,'Recupera finalidad y día con perspectiva mi en lugar de tu.'),('F',0,'Fragmento sin identificar información conocida y desconocida.')],
 [('F',0,'Fragmento sin recuperar pasear como ayuda.'),('F',0,'No identifica pasear antes de escribir.'),('F',1,'Niega el anochecer después de la corrección explícita.'),('F',0,'Fragmentos repetidos sin relacionar paseo y escritura.')]
],28)
