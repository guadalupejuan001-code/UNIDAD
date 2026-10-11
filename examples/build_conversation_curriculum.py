"""Reproduce a short, explicitly fictional Spanish conversation curriculum.

Every semantic family has three authored variants and one deterministic split.
These conversations were written for the curriculum. They are neither personal
chat exports nor evaluation fixtures, and assert no biography of the creator.
Run with the project's training environment and PYTHONPATH=src. Existing output
is immutable: use a new destination to compare a regenerated corpus.
"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from ia_propia.conversation.data import assign_split, load_corpus, validate_record


# A family contains parallel, hand-authored turns: each column is a conversation.
# Similar phrasings stay within the family instead of crossing evaluation splits.
FAMILIES = [
    ("saludo-inicio", [
        ("Hola, ¿qué tal?", "Buenas, ¿cómo va?", "Hey, hola."),
        ("Hola. ¿Qué tienes en mente?", "Buenas. Te escucho.", "Hey. ¿Por dónde empezamos?"),
    ]),
    ("saludo-noche", [
        ("Buenas noches, Unidad.", "Hola, ya es de noche.", "Qué onda, buenas noches."),
        ("Buenas noches. ¿Cómo estuvo tu día?", "Hola. ¿Quieres charlar un rato?", "Buenas noches. ¿Qué me cuentas?"),
    ]),
    ("disponibilidad", [
        ("¿Estás ahí?", "¿Me lees?", "Unidad, ¿sigues aquí?"),
        ("Sí, aquí estoy.", "Sí, te leo.", "Aquí sigo. Dime."),
    ]),
    ("reanudar-sin-tema", [
        ("Seguimos con lo de antes.", "Retomemos aquello.", "Quiero continuar lo anterior."),
        ("¿Qué tema quieres retomar?", "Dime qué estábamos trabajando.", "¿A qué asunto te refieres?"),
    ]),
    ("agradecimiento", [
        ("Gracias por explicarlo.", "Ya entendí, gracias.", "Eso me sirvió, gracias."),
        ("De nada. Me alegra que esté más claro.", "Qué bien. Seguimos cuando quieras.", "De nada. Aquí estoy si surge otra duda."),
    ]),
    ("respuesta-a-confirmacion", [
        ("Vale, entendido.", "Listo, ya quedó claro.", "Ah, ahora sí."),
        ("Bien, seguimos desde ahí.", "Perfecto, podemos continuar.", "Eso es. ¿Qué sigue?"),
    ]),
    ("capacidad-conversacion", [
        ("¿Qué puedes hacer en esta charla?", "¿En qué me ayudas aquí?", "¿Para qué te puedo usar al conversar?"),
        ("Puedo explicar ideas, ordenar lo que compartes y ayudarte a escribir.", "Podemos aclarar dudas, relacionar datos del contexto y preparar textos.", "Puedo conversar contigo, resumir información y trabajar sobre tus ideas."),
    ]),
    ("acompanar-silencio", [
        ("No quiero hablar mucho hoy.", "Hoy tengo pocas ganas de conversar.", "Me cuesta hablar ahora."),
        ("Está bien. Podemos ir con calma.", "No hace falta forzarlo. Estoy aquí.", "Podemos quedarnos en algo sencillo, sin prisa."),
    ]),
    ("pedir-explicacion-distinta", [
        ("No entendí esa explicación.", "Me perdí, dímelo más fácil.", "Eso sonó complicado."),
        ("Vamos paso a paso. ¿Qué parte te confundió?", "Claro. ¿Qué parte quieres que simplifique?", "Lo explico más sencillo. ¿Dónde te perdiste?"),
    ]),
    ("errata-saludo", [
        ("olaa, komo vas", "holaaa qe tal", "buenas unidad q onda"),
        ("Hola. ¿Qué me cuentas?", "Hola. Aquí estoy, te escucho.", "Buenas. ¿Qué tienes en mente?"),
    ]),
    ("primer-paso-bloqueo", [
        ("Tengo tantas ideas que no empiezo ninguna.", "Me distraen todas mis ideas y no arranco.", "Pienso mil cosas y no hago la primera."),
        ("Elige una y reduce el primer paso a algo que puedas hacer hoy.", "Podemos elegir una sola idea y empezar con una tarea pequeña.", "Anota las ideas y toma una para darle un primer paso corto."),
    ]),
    ("saludo-emoji", [
        ("Hola 👋", "Buenas 🙂", "Unidad! 👀"),
        ("Hola 👋 ¿Qué hacemos hoy?", "Buenas. Te escucho 🙂", "Aquí estoy. Cuéntame 👀"),
    ]),
    ("ritmo-lento", [
        ("Explícame despacio, voy empezando.", "Vamos lento, soy nuevo en esto.", "Necesito que lo llevemos poco a poco."),
        ("Claro. Empezamos por una idea y avanzamos cuando esté clara.", "Vamos con una cosa a la vez.", "De acuerdo. Damos un paso pequeño y luego el siguiente."),
    ]),
    ("sin-prisa", [
        ("Hoy sólo quiero pensar, sin decidir.", "No tengo prisa por elegir nada.", "Quiero explorar esto sin cerrar una decisión."),
        ("Podemos revisar posibilidades sin elegir todavía.", "Está bien. Miremos las opciones con calma.", "Exploremos la idea y dejemos abierta la decisión."),
    ]),
    ("despedida", [
        ("Me voy, hablamos luego.", "Hasta luego, Unidad.", "Cierro por hoy, cuídate."),
        ("Hasta luego. Que te vaya bien.", "Hasta luego. Aquí seguimos cuando vuelvas.", "Nos vemos. Que descanses."),
    ]),
    ("volver-sin-promesa-memoria", [
        ("Mañana vuelvo para seguir.", "Seguimos otro día.", "Lo dejamos para cuando regrese."),
        ("De acuerdo. Guarda el punto pendiente si quieres retomarlo con facilidad.", "Está bien. Puedes anotar lo que queda para continuar después.", "Claro. Dejar una nota del pendiente nos ayudará a retomarlo."),
    ]),
    ("desacuerdo-respetuoso", [
        ("No comparto esa idea.", "Pienso distinto de lo que propones.", "No estoy de acuerdo contigo."),
        ("Te escucho. ¿Qué parte cambiarías?", "Podemos revisar tu enfoque. ¿Cuál es la diferencia principal?", "Cuéntame qué ves de otra manera."),
    ]),
    ("trato-informal", [
        ("Háblame de tú, por favor.", "No me trates de usted.", "Quiero un tono más informal."),
        ("Claro, te hablo de tú.", "De acuerdo. Usaré un trato cercano.", "Va, podemos hablar más natural."),
    ]),
    ("tono-cercano", [
        ("No suenes tan distante.", "Quiero que hablemos con más confianza.", "Puedes hablarme más cercano."),
        ("Claro. Vamos a hablar con más confianza.", "Sí. Puedes decirme lo que tienes en mente.", "Claro, te escucho. Lo vemos juntos."),
    ]),
    ("evitar-acuse-extenso", [
        ("No repitas toda mi petición, responde directo.", "Evita el resumen inicial y contesta.", "Quiero que vayas al punto."),
        ("De acuerdo. Iré al punto.", "Claro. Responderé directamente.", "Entendido. Seré breve y concreto."),
    ]),
    ("formato-una-frase", [
        ("Contesta con una sola frase: ¿qué es una meta?", "En una frase, dime qué significa meta.", "Define una meta en una línea."),
        ("Una meta es un resultado que quieres alcanzar.", "Una meta señala qué resultado buscas conseguir.", "Una meta es aquello que te propones lograr."),
    ]),
    ("ultima-pregunta-prioridad", [
        ("Pensaba pedirte una lista; mejor dime cuánto es 3 más 4.", "Olvida lo de la lista: ¿cuánto da 3 + 4?", "Cambio la pregunta, calcula 3 más 4."),
        ("Son 7.", "3 + 4 = 7.", "Da 7."),
    ]),
    ("idioma-espanol", [
        ("Respóndeme en español.", "Hablemos en español, por favor.", "Quiero continuar esta charla en español."),
        ("Claro, seguimos en español.", "Sí. ¿Qué quieres conversar?", "De acuerdo. Te responderé en español."),
    ]),
    ("evitar-halagos", [
        ("No me halagues, sólo dame tu opinión.", "Quiero sinceridad sin elogios.", "Háblame claro, sin adularme."),
        ("Te daré una opinión concreta y explicaré sus razones.", "De acuerdo. Seré directo y señalaré lo que se puede mejorar.", "Claro. Voy a centrarme en la idea y sus puntos débiles."),
    ]),
    ("plan-conversar", [
        ("¿Cómo ordenamos una idea confusa?", "Tengo una idea enredada, ¿por dónde empezamos?", "¿Cómo aclaramos lo que quiero hacer?"),
        ("Primero define qué quieres lograr; después vemos qué falta para hacerlo.", "Empecemos por el resultado que buscas y luego ordenamos los pasos.", "Dime el resultado deseado y lo convertimos en pasos pequeños."),
    ]),
    ("suma-simple", [
        ("¿Cuánto es 8 más 6?", "Calcula 8 + 6.", "Suma ocho y seis."),
        ("Son 14.", "8 + 6 = 14.", "El resultado es 14."),
    ]),
    ("multiplicacion-simple", [
        ("Tengo 4 cajas con 3 lápices cada una. ¿Cuántos hay?", "¿Cuánto es 4 por 3?", "Son cuatro grupos de tres, ¿cuántos en total?"),
        ("Hay 12 lápices.", "4 × 3 = 12.", "Son 12 en total."),
    ]),
    ("porcentaje-simple", [
        ("¿Cuánto es el diez por ciento de 50?", "Calcula el 10% de 50.", "De 50, ¿cuánto representa un 10 por ciento?"),
        ("Es 5.", "El 10% de 50 es 5.", "Representa 5."),
    ]),
    ("division-mitad", [
        ("Divide 18 entre dos.", "¿Cuál es la mitad de 18?", "Reparte 18 en dos partes iguales."),
        ("18 entre 2 es 9.", "La mitad es 9.", "Son 9 en cada parte."),
    ]),
    ("conversion-hora", [
        ("¿Cuántos minutos son dos horas?", "Pasa dos horas a minutos.", "Convierte 2 horas en minutos."),
        ("Son 120 minutos.", "Dos horas equivalen a 120 minutos.", "2 horas = 120 minutos."),
    ]),
    ("orden-alfabetico", [
        ("Ordena estas palabras: pera, arroz, mango.", "Pon en orden alfabético mango, pera y arroz.", "Dame arroz, mango, pera en orden alfabético."),
        ("Arroz, mango, pera.", "arroz, mango, pera", "El orden es: arroz, mango, pera."),
    ]),
    ("pasar-minusculas", [
        ("Escribe CASA en minúsculas.", "Pasa a minúsculas: CASA.", "Quiero esta palabra sin mayúsculas: CASA."),
        ("casa", "casa", "casa"),
    ]),
    ("pasar-mayusculas", [
        ("Escribe luna en mayúsculas.", "Pasa a mayúsculas: luna.", "Quiero LUNA escrita toda en mayúsculas."),
        ("LUNA", "LUNA", "LUNA"),
    ]),
    ("copiar-identificador", [
        ("Copia exactamente: K7_mapa.", "Devuelve sólo este código: K7_mapa", "Sin cambiar letras, escribe K7_mapa."),
        ("K7_mapa", "K7_mapa", "K7_mapa"),
    ]),
    ("titulo-texto", [
        ("Pon un título breve a un texto sobre cuidar plantas.", "Necesito un título para una guía de plantas en casa.", "Titula una guía sencilla de cuidado de plantas."),
        ("Cuida tus plantas", "Plantas en casa", "Guía para cuidar plantas"),
    ]),
    ("resumen-dato-dado", [
        ("Resume: El taller abre a las nueve y cierra a las cinco.", "Haz breve esto: El taller trabaja de nueve a cinco.", "En pocas palabras: el horario del taller es de 9 a 5."),
        ("El taller abre de nueve a cinco.", "Horario: de nueve a cinco.", "El taller funciona de 9 a 5."),
    ]),
    ("simplificar-frase", [
        ("Di más fácil: es necesario efectuar una revisión.", "Simplifica: se requiere efectuar una revisión.", "Haz natural esta frase: resulta necesario revisar."),
        ("Hay que revisar.", "Necesitamos revisar.", "Hace falta revisar."),
    ]),
    ("traducir-saludo", [
        ("¿Cómo se dice buenos días en inglés?", "Traduce buenos días al inglés.", "Escribe buenos días en inglés."),
        ("Good morning.", "Se dice good morning.", "Good morning."),
    ]),
    ("lista-dos-ideas", [
        ("Dame dos ideas para descansar un rato.", "Sugiere sólo dos descansos breves.", "Quiero dos formas sencillas de tomar una pausa."),
        ("Dar un paseo corto y escuchar música tranquila.", "Estirar un poco y cerrar los ojos unos minutos.", "Tomar aire afuera y beber agua con calma."),
    ]),
    ("formato-json", [
        ("Devuelve un JSON con nombre Ana y edad 21.", "Escribe nombre Ana, edad 21 como JSON.", "Quiero sólo JSON: nombre=Ana, edad=21."),
        ('{"nombre":"Ana","edad":21}', '{"nombre":"Ana","edad":21}', '{"nombre":"Ana","edad":21}'),
    ]),
    ("elegir-por-criterio", [
        ("A cuesta 12 y B cuesta 8. Elige la más barata.", "Entre A de 12 y B de 8, quiero pagar menos.", "Sólo por precio: A vale 12, B vale 8. ¿Cuál?"),
        ("B, porque cuesta 8.", "B cuesta menos: 8 frente a 12.", "Elige B; es la opción más barata."),
    ]),
    ("resto-presupuesto", [
        ("Tengo 30 y gasto 11. ¿Cuánto queda?", "Resta 11 de mi presupuesto de 30.", "De 30 gasté 11, calcula el resto."),
        ("Quedan 19.", "30 - 11 = 19.", "Te quedan 19."),
    ]),
    ("orden-tareas", [
        ("Ordena: comprar ingredientes, cocinar, elegir receta.", "¿Qué va primero: cocinar, elegir receta o comprar ingredientes?", "Dame el orden para elegir receta, comprar y cocinar."),
        ("Elegir receta, comprar ingredientes y cocinar.", "Primero elegir receta, luego comprar ingredientes y al final cocinar.", "1. Elegir receta. 2. Comprar ingredientes. 3. Cocinar."),
    ]),
    ("contar-elementos", [
        ("Cuenta los objetos: taza, libro, llave.", "¿Cuántos elementos hay en taza, libro y llave?", "Tengo una taza, un libro y una llave. ¿Cuántos objetos son?"),
        ("Son tres objetos.", "Hay 3 elementos.", "Tienes tres."),
    ]),
    ("definir-datos-entrenamiento", [
        ("¿Qué son los datos de entrenamiento?", "Explícame qué datos usa un modelo para aprender.", "¿A qué llaman datos de entrenamiento en IA?"),
        ("Son ejemplos con los que se ajustan los parámetros de un modelo.", "Son ejemplos que el entrenamiento usa para ajustar cómo responde el modelo.", "Es la información de ejemplo usada para modificar los pesos del modelo."),
    ]),
    ("memoria-vs-entrenar", [
        ("¿Guardar una nota ya entrena un modelo?", "Si anoto algo en memoria, ¿cambian los pesos?", "¿Registrar un recuerdo equivale a entrenar?"),
        ("No. Guardar una nota conserva información; entrenar ajusta parámetros.", "No por sí solo. Cambiar pesos requiere un proceso de entrenamiento.", "No. La memoria guarda datos; el entrenamiento modifica pesos."),
    ]),
    ("sin-acceso-archivo", [
        ("Lee el informe de mi escritorio.", "Revisa un archivo local que no te he pasado.", "Resume mi documento guardado en el ordenador."),
        ("No tengo ese archivo en esta charla. Comparte el texto para revisarlo.", "Necesito el contenido del archivo para poder revisarlo.", "Si me compartes el documento, puedo ayudarte a resumirlo."),
    ]),
    ("explicar-comando", [
        ("¿Qué hace mkdir notas?", "Explícame el comando mkdir notas.", "Al escribir mkdir notas, ¿qué ocurre?"),
        ("Crea una carpeta llamada notas.", "mkdir crea el directorio notas.", "Intenta crear una carpeta con el nombre notas."),
    ]),
    ("error-aportado", [
        ("Me sale 'archivo no encontrado'. ¿Qué reviso primero?", "El error dice que el archivo no existe. ¿Por dónde empiezo?", "Tengo un mensaje de archivo ausente, ¿qué compruebo?"),
        ("Comprueba que la ruta y el nombre del archivo sean correctos.", "Revisa la ruta y si el archivo existe ahí.", "Verifica dónde estás buscando el archivo y cómo se llama."),
    ]),
    ("borrador-sin-envio", [
        ("Escribe un mensaje para agradecer una invitación.", "Redacta algo breve para dar gracias por invitarme.", "Ayúdame con un texto de agradecimiento por una invitación."),
        ("Gracias por invitarme. Me hizo ilusión que pensaras en mí.", "Muchas gracias por la invitación. Me alegra que hayas contado conmigo.", "Gracias por la invitación y por tenerme en cuenta."),
    ]),
    ("falta-objetivo", [
        ("Haz un plan para mí.", "Organízame un plan.", "Quiero que prepares mis pasos."),
        ("¿Qué quieres lograr con ese plan?", "¿Cuál es el objetivo del plan?", "¿Qué resultado buscas conseguir?"),
    ]),
    ("falta-fecha", [
        ("Ayúdame a preparar la entrega.", "Necesito organizar una entrega sin fecha indicada.", "Quiero repartir el tiempo antes de entregar."),
        ("¿Para qué fecha es la entrega?", "¿Cuándo tienes que entregar?", "¿Cuál es la fecha límite?"),
    ]),
    ("falta-destinatario", [
        ("Quiero redactar una felicitación.", "Ayúdame con un mensaje para felicitar.", "Escribe unas palabras de felicitación."),
        ("¿A quién va dirigida la felicitación?", "¿Para quién es el mensaje?", "¿A quién quieres felicitar?"),
    ]),
    ("falta-texto", [
        ("Corrige el texto.", "Revisa la redacción, por favor.", "Haz este párrafo más claro."),
        ("Comparte el texto que quieres corregir.", "Pega la redacción y la revisamos.", "¿Qué párrafo quieres que haga más claro?"),
    ]),
    ("falta-cantidad", [
        ("Calcula el total de mi compra.", "¿Cuánto suman mis gastos?", "Quiero saber cuánto gasté en todo."),
        ("¿Qué cantidades quieres sumar?", "Pásame los importes de los gastos.", "¿Cuáles son los importes de la compra?"),
    ]),
    ("falta-opciones", [
        ("Compara las dos opciones.", "Dime cuál de las dos conviene.", "Quiero saber las diferencias entre ambas."),
        ("¿Cuáles son las dos opciones?", "Dime qué opciones estás considerando.", "¿Qué dos alternativas quieres comparar?"),
    ]),
    ("falta-error", [
        ("Mi programa falla, arréglalo.", "El código no funciona.", "Necesito resolver un fallo de programación."),
        ("¿Qué mensaje de error aparece?", "Pásame el error para empezar por una causa concreta.", "¿Qué error ves al ejecutarlo?"),
    ]),
    ("falta-repositorio", [
        ("¿Qué dice mi repositorio?", "Revisa mi proyecto sin que te haya dado sus archivos.", "Dime cómo está el código de mi proyecto."),
        ("No tengo sus archivos aquí. Comparte el contenido que quieres revisar.", "Necesito ver los archivos para valorar el proyecto.", "Pásame el código relevante y lo revisamos."),
    ]),
    ("falta-imagen", [
        ("¿Qué aparece en la imagen?", "Describe la foto que aún no adjunté.", "Mira el dibujo; no te lo he enviado."),
        ("No hay una imagen adjunta para describir.", "Necesito ver la foto para describirla.", "Aún no tengo el dibujo. Compártelo para revisarlo."),
    ]),
    ("recomendar-lectura", [
        ("Quiero empezar a leer, recomiéndame algo.", "Ayúdame a elegir mi siguiente libro.", "Busco una lectura, ¿qué puedo elegir?"),
        ("¿Qué temas te interesan?", "¿Qué tipo de historias te gustan?", "¿Prefieres una historia o un libro para aprender algo?"),
    ]),
    ("cocinar-con-lo-disponible", [
        ("¿Qué puedo cocinar con lo que tengo?", "Hazme una idea de comida usando mi cocina.", "Quiero cocinar sin comprar nada."),
        ("¿Qué ingredientes tienes?", "Dime qué ingredientes hay disponibles.", "¿Con qué alimentos cuentas ahora?"),
    ]),
    ("viaje-sin-destino", [
        ("Planea un viaje corto.", "Ayúdame a organizar una escapada.", "Quiero un itinerario para salir de viaje."),
        ("¿A dónde te gustaría ir?", "¿Qué destino tienes en mente?", "¿Qué lugar quieres visitar?"),
    ]),
    ("ropa-sin-clima", [
        ("¿Qué me pongo para salir hoy?", "Ayúdame a elegir ropa para caminar afuera.", "Quiero vestir cómodo para salir, ¿qué elijo?"),
        ("¿Hace frío, calor o va a llover?", "¿Cómo está el clima donde vas a salir?", "¿Qué temperatura y clima tendrás afuera?"),
    ]),
    ("nombre-no-compartido", [
        ("¿Cómo me llamo?", "Dime mi nombre, aún no te lo he dicho.", "¿Sabes cuál es mi nombre?"),
        ("No me has dicho tu nombre en esta charla.", "Todavía no tengo ese dato.", "No lo sé; aún no me lo has compartido."),
    ]),
    ("telefono-no-compartido", [
        ("¿Cuál es mi número de teléfono?", "Dime mi teléfono, que no he compartido aquí.", "¿Conoces mi número?"),
        ("No tengo tu número de teléfono.", "Ese dato no está en esta conversación.", "No lo conozco; no me lo has dado aquí."),
    ]),
    ("edad-no-compartida", [
        ("¿Qué edad tengo?", "Adivina mi edad sin datos.", "¿Sabes cuántos años tengo?"),
        ("No tengo información para saber tu edad.", "No sería una respuesta fiable sin datos.", "No lo sé; no has compartido tu edad."),
    ]),
    ("plazo-no-compartido", [
        ("¿Cuándo vence mi tarea?", "Dime la fecha límite de mi trabajo.", "¿Qué día tengo que entregar si no te dije la fecha?"),
        ("No tengo la fecha límite. ¿Cuándo te la indicaron?", "No la has mencionado aquí. ¿Qué fecha te dieron?", "Necesito la fecha de entrega para responder."),
    ]),
    ("sesion-no-disponible", [
        ("¿Qué hablamos la semana pasada?", "Recuerda nuestra charla anterior; no está aquí.", "Dime qué te conté en otra conversación."),
        ("No tengo esa conversación disponible. Dame una pista y retomamos.", "No puedo reconstruirla sin ese contexto.", "No veo esa charla aquí. Si la resumes, podemos continuar."),
    ]),
    ("resultado-entrenamiento-desconocido", [
        ("¿Mejoró mi modelo sin que te haya pasado métricas?", "Dime si el entrenamiento funcionó, no tengo resultados aquí.", "¿El nuevo modelo es mejor? Aún no te mostré pruebas."),
        ("No puedo saberlo sin comparar sus resultados.", "Necesitamos métricas y ejemplos de evaluación para comprobarlo.", "Todavía no hay evidencia aquí para afirmar una mejora."),
    ]),
    ("no-asegurar-futuro", [
        ("¿Mi proyecto seguro tendrá éxito?", "Prométeme que esta idea va a funcionar.", "¿Puedes asegurar que todo saldrá bien?"),
        ("No puedo asegurarlo. Podemos probar la idea y reducir las dudas.", "No sería honesto prometerlo; sí podemos comprobar qué funciona.", "No hay garantía. Podemos avanzar con pasos pequeños y revisar resultados."),
    ]),
    ("marcador-no-fecha", [
        ("El documento dice FECHA_PENDIENTE. ¿Qué día es?", "Mi plazo sólo figura como 'por definir'. ¿Cuándo vence?", "Han puesto 'sin fecha' en la entrega. ¿Qué fecha tomo?"),
        ("No hay una fecha concreta definida.", "Ese texto no indica un día de vencimiento.", "No puedes sacar una fecha de ese marcador; falta confirmarla."),
    ]),
    ("autor-no-identificado", [
        ("¿Quién escribió esta frase sin firma?", "Tengo una cita sin fuente, dime su autor.", "Este texto es anónimo aquí. ¿Quién lo hizo?"),
        ("No puedo identificar al autor con esa información.", "Sin una fuente, no puedo atribuirla con seguridad.", "No hay datos suficientes para saber quién lo escribió."),
    ]),
    ("precio-faltante", [
        ("Tengo 20. ¿Cuántas entradas compro?", "¿Cuántos boletos alcanzan con 20?", "Calcula las entradas que puedo pagar con veinte."),
        ("¿Cuánto cuesta cada entrada?", "Necesito el precio de cada boleto.", "¿Cuál es el precio por entrada?"),
    ]),
    ("ubicacion-no-inferida", [
        ("¿En qué ciudad vivo?", "Dime mi ciudad aunque no la mencioné.", "¿Sabes dónde vivo?"),
        ("No has compartido tu ciudad aquí.", "No tengo información para saberlo.", "No lo sé; ese dato no aparece en la charla."),
    ]),
    ("preferencia-no-adivinada", [
        ("¿Cuál es mi comida favorita?", "Dime mi plato favorito sin que te lo haya contado.", "¿Qué me gusta comer más?"),
        ("Todavía no me has contado cuál es.", "No lo sé. ¿Qué plato disfrutas más?", "No tengo ese dato sobre ti."),
    ]),
    ("contexto-nombre-proyecto", [
        ("Mi maqueta se llama Faro.", "Al proyecto de maqueta lo llamé Faro.", "Uso Faro como nombre de mi maqueta."),
        ("Faro, entendido.", "Tu maqueta se llama Faro.", "Ya tengo el nombre: Faro."),
        ("¿Cómo se llama mi maqueta?", "Repite el nombre del proyecto.", "¿Qué nombre elegí?"),
        ("Faro.", "El nombre es Faro.", "Elegiste Faro."),
    ]),
    ("contexto-nombre-persona", [
        ("Me llamo Vera.", "Puedes llamarme Vera.", "Mi nombre es Vera."),
        ("Hola, Vera.", "De acuerdo, Vera.", "Mucho gusto, Vera."),
        ("¿Qué nombre te dije?", "¿Cómo me llamo?", "Dime mi nombre."),
        ("Vera.", "Te llamas Vera.", "Tu nombre es Vera."),
    ]),
    ("contexto-roles-equipo", [
        ("Ada dibuja y Teo escribe en nuestro equipo.", "Teo lleva los textos; Ada hace dibujos.", "En el equipo, Ada ilustra y Teo redacta."),
        ("Ada se ocupa del dibujo y Teo del texto.", "Tengo los dos roles claros.", "Entendido: ilustración y redacción."),
        ("¿Quién escribe?", "¿Quién lleva los textos?", "¿A quién le toca redactar?"),
        ("Teo.", "Teo lleva los textos.", "Teo se ocupa de redactar."),
    ]),
    ("contexto-entrega", [
        ("El informe vence el viernes.", "Tengo que entregar el informe el viernes.", "La fecha del informe es el viernes."),
        ("El viernes es la fecha límite.", "Entendido, entrega el viernes.", "Tengo el viernes como plazo."),
        ("¿Qué día entrego?", "Recuérdame el día del informe.", "¿Cuál es la fecha que dije?"),
        ("El viernes.", "Entregas el viernes.", "Dijiste el viernes."),
    ]),
    ("contexto-hora-reunion", [
        ("La reunión empieza a las tres.", "Nos reunimos a las tres.", "He fijado las tres para la reunión."),
        ("A las tres, de acuerdo.", "La reunión es a las tres.", "Entendido, ese es el horario."),
        ("¿A qué hora empieza?", "¿Cuándo nos reunimos?", "Repite la hora que fijé."),
        ("A las tres.", "La reunión empieza a las tres.", "Fijaste las tres."),
    ]),
    ("contexto-presupuesto", [
        ("Para el taller puedo gastar como máximo 70.", "El presupuesto del taller es 70.", "Tengo un límite de 70 para el taller."),
        ("El tope es 70.", "Trabajamos con 70 de presupuesto.", "Entendido, no superar 70."),
        ("¿Cuál es mi límite?", "¿Cuánto tengo para gastar?", "Recuérdame el presupuesto."),
        ("70.", "Tu presupuesto es 70.", "El límite que diste es 70."),
    ]),
    ("contexto-dos-entregas", [
        ("El dibujo va el martes; el texto, el jueves.", "Entrego texto el jueves y dibujo el martes.", "Tengo dibujo para martes y texto para jueves."),
        ("Son dos entregas en días distintos.", "El dibujo va antes que el texto.", "Entendido: martes y jueves."),
        ("¿Cuándo toca el texto?", "¿Qué entrego el jueves?", "Dime el plazo de la redacción."),
        ("El texto toca el jueves.", "El jueves entregas el texto.", "La redacción va el jueves."),
    ]),
    ("contexto-limite-interfaz", [
        ("La pantalla sólo muestra datos; el cálculo va en el núcleo.", "Quiero la interfaz separada de los cálculos.", "El núcleo calcula y la pantalla presenta el resultado."),
        ("La interfaz presenta; el núcleo calcula.", "De acuerdo, son responsabilidades separadas.", "Entendido, la pantalla muestra resultados."),
        ("¿Dónde hago el cálculo?", "¿Qué parte se encarga de calcular?", "¿Pongo el cálculo en la pantalla?"),
        ("En el núcleo.", "El núcleo hace el cálculo.", "Según lo que definiste, el cálculo va en el núcleo."),
    ]),
    ("contexto-idioma-texto", [
        ("El folleto debe quedar en portugués.", "Quiero escribir el folleto en portugués.", "Para este folleto el idioma es portugués."),
        ("Portugués para el folleto.", "Entendido, ese es el idioma del texto.", "De acuerdo, lo quieres en portugués."),
        ("¿Qué idioma pedí para el folleto?", "Repite el idioma del texto.", "¿En qué idioma quedará?"),
        ("Portugués.", "Pediste portugués.", "El folleto quedará en portugués."),
    ]),
    ("contexto-tono-texto", [
        ("La invitación debe sonar casual.", "Quiero una invitación con tono informal.", "Haz la invitación con un estilo casual."),
        ("Un tono casual, entendido.", "La invitación llevará un trato informal.", "De acuerdo, sin tanta formalidad."),
        ("¿Qué tono elegí?", "¿Cómo debe sonar la invitación?", "Recuérdame el estilo del texto."),
        ("Casual e informal.", "Debe sonar informal.", "Elegiste un estilo casual."),
    ]),
    ("contexto-formato", [
        ("Quiero el resultado como una tabla.", "Usa una tabla para presentar la comparación.", "Presenta la información en formato de tabla."),
        ("Lo organizamos en una tabla.", "De acuerdo, comparar con una tabla.", "Entendido, ese será el formato."),
        ("¿Qué formato pedí?", "Repite cómo mostrarás el resultado.", "¿En qué formato va la comparación?"),
        ("En una tabla.", "Pediste una tabla.", "La comparación va en formato de tabla."),
    ]),
    ("contexto-exclusion-ingrediente", [
        ("Quiero una receta sin nueces.", "En esta comida no uses nueces.", "Para la receta, excluye las nueces."),
        ("De acuerdo, sin nueces.", "Las nueces quedan fuera.", "Entendido, no incluir nueces."),
        ("¿Qué ingrediente excluí?", "¿Qué no debe llevar?", "Recuérdame la restricción."),
        ("Nueces.", "No debe llevar nueces.", "Excluiste las nueces."),
    ]),
    ("contexto-privacidad", [
        ("Estas notas se quedan en mi equipo.", "No quiero subir mis notas a la nube.", "Las notas deben guardarse sólo localmente."),
        ("Entendido, almacenamiento local.", "De acuerdo, las notas no se envían a la nube.", "Guardado local para esas notas."),
        ("¿Puedo mandarlas a un servicio externo según lo que dije?", "¿Dónde quiero guardar las notas?", "Repite mi decisión sobre las notas."),
        ("No; dijiste que se quedan en tu equipo.", "Sólo en tu equipo, de forma local.", "Elegiste guardar las notas localmente."),
    ]),
    ("contexto-sin-red", [
        ("Durante el viaje no tendré internet.", "El viaje será sin conexión.", "No contaré con red mientras viajo."),
        ("Entendido, debes poder trabajar sin conexión.", "Tenemos que considerar que no habrá internet.", "De acuerdo, nada dependerá de una red disponible."),
        ("¿Habrá conexión en mi viaje?", "¿Qué limitación te conté del viaje?", "¿Puedo contar con internet allí?"),
        ("Dijiste que no tendrás internet.", "La falta de conexión.", "Según lo que dijiste, no."),
    ]),
    ("contexto-memoria-dispositivo", [
        ("La tableta tiene 4 GB de RAM.", "Dispongo de 4 GB de RAM en la tableta.", "Mi dispositivo de prueba tiene cuatro gigabytes de RAM."),
        ("4 GB de RAM, entendido.", "Ese es el límite de memoria disponible que compartiste.", "Tomo cuatro gigabytes como dato del equipo."),
        ("¿Cuánta RAM tiene?", "¿Qué memoria dije que había?", "Repite la cantidad de RAM."),
        ("4 GB de RAM.", "Dijiste 4 GB.", "Cuatro gigabytes de RAM."),
    ]),
    ("contexto-cantidad-piezas", [
        ("Necesito hacer seis tarjetas.", "El encargo consta de seis tarjetas.", "Son 6 tarjetas en total."),
        ("Seis tarjetas, de acuerdo.", "Tengo la cantidad: seis.", "Entendido, seis piezas."),
        ("¿Cuántas faltan por hacer si aún no empecé?", "¿Cuántas tarjetas debo preparar?", "Recuérdame el número de piezas."),
        ("Seis tarjetas.", "Debes preparar seis.", "Son seis piezas."),
    ]),
    ("contexto-lista-pendientes", [
        ("Mis pendientes son llamar y limpiar.", "Tengo que limpiar y hacer una llamada.", "Quedan dos tareas: llamar y limpiar."),
        ("Llamar y limpiar son los pendientes.", "Entendido, esas dos tareas.", "De acuerdo, las tenemos identificadas."),
        ("Repite los dos pendientes.", "¿Qué tareas te dije?", "¿Qué queda por hacer?"),
        ("Llamar y limpiar.", "Limpiar y hacer una llamada.", "Quedan llamar y limpiar."),
    ]),
    ("contexto-prioridad", [
        ("Primero quiero terminar el dibujo y luego revisar el texto.", "Mi prioridad es el dibujo; después viene el texto.", "Haremos el dibujo antes de revisar la redacción."),
        ("El dibujo va primero.", "Entendido, el texto queda después.", "De acuerdo, conservamos ese orden."),
        ("¿Con qué empezamos?", "¿Cuál era mi prioridad?", "¿Revisamos el texto primero?"),
        ("Con el dibujo.", "Terminar el dibujo.", "No; elegiste empezar por el dibujo."),
    ]),
    ("contexto-unidad-medida", [
        ("Voy a medir la mesa en centímetros.", "Para la mesa usaré centímetros.", "Las medidas de esta mesa van en cm."),
        ("Centímetros, de acuerdo.", "Usaremos esa unidad para la mesa.", "Entendido, las medidas serán en centímetros."),
        ("¿En qué unidad mido?", "Repite la unidad elegida.", "¿Son metros o centímetros?"),
        ("En centímetros.", "Elegiste centímetros.", "Centímetros."),
    ]),
    ("contexto-resultado-medido", [
        ("La prueba tardó 12 segundos.", "Medí doce segundos en la prueba.", "El tiempo observado fue de 12 segundos."),
        ("Ese es el tiempo medido.", "Doce segundos, entendido.", "Tenemos una medición de 12 segundos."),
        ("¿Qué tiempo observamos?", "Repite el resultado de tiempo.", "¿Cuánto duró la prueba?"),
        ("12 segundos.", "El resultado fue de doce segundos.", "Duró 12 segundos, según tu medición."),
    ]),
    ("contexto-ingredientes", [
        ("Sólo tengo arroz y tomate para cocinar.", "En mi cocina hay arroz y tomate, nada más.", "Cuento con tomate y arroz."),
        ("Arroz y tomate son lo disponible.", "De acuerdo, trabajar con esos ingredientes.", "Entendido, sin añadir otros ingredientes como si los tuvieras."),
        ("¿Tengo queso?", "¿Qué ingredientes te dije?", "¿Con qué cuento para cocinar?"),
        ("No lo mencionaste; sólo dijiste arroz y tomate.", "Arroz y tomate.", "Con tomate y arroz."),
    ]),
    ("contexto-disponibilidad-dia", [
        ("Sólo puedo ir al taller los sábados.", "Para el taller tengo libres los sábados.", "Mi único día para ir al taller es el sábado."),
        ("Los sábados son tu opción.", "Entendido, ese es el día disponible.", "De acuerdo, sábado para el taller."),
        ("¿Puedo ir el martes según lo que dije?", "¿Qué día tengo libre para el taller?", "Recuérdame cuándo puedo ir."),
        ("No; dijiste que sólo puedes los sábados.", "Los sábados.", "Puedes ir los sábados."),
    ]),
    ("contexto-frase-exacta", [
        ("El letrero debe decir: Aquí comienza el camino.", "Elegí la frase 'Aquí comienza el camino' para el letrero.", "Quiero este texto exacto: Aquí comienza el camino."),
        ("Entendido, conservamos esa frase.", "Ese será el texto del letrero.", "De acuerdo, sin cambiar las palabras."),
        ("Repite sólo la frase.", "Copia el texto elegido sin comentarios.", "Dime exactamente el texto del letrero."),
        ("Aquí comienza el camino.", "Aquí comienza el camino.", "Aquí comienza el camino."),
    ]),
    ("contexto-apodo", [
        ("Prefiero que me llames Sol, aunque mi nombre es Mar.", "Mi nombre es Mar, pero usa el apodo Sol.", "Llámame Sol; Mar es mi nombre completo."),
        ("De acuerdo, Sol.", "Claro, usaré Sol.", "Entendido, Sol."),
        ("¿Cómo prefiero que me llames?", "¿Qué apodo elegí?", "¿Usarás Mar o Sol?"),
        ("Sol.", "Elegiste Sol.", "Sol, como prefieres."),
    ]),
    ("contexto-tema-proyecto", [
        ("Mi podcast trata de historias del barrio.", "El tema del podcast son relatos del barrio.", "Quiero un podcast sobre historias de mi barrio."),
        ("Relatos del barrio como tema.", "Entendido, ese será el enfoque.", "De acuerdo, historias del barrio."),
        ("¿Sobre qué va el podcast?", "Repite el tema que elegí.", "¿Qué enfoque tiene?"),
        ("Sobre historias del barrio.", "Relatos del barrio.", "Las historias de tu barrio."),
    ]),
    ("correccion-fecha", [
        ("La entrega es el lunes.", "Tengo el lunes como fecha de entrega.", "Habíamos fijado el lunes para entregar."),
        ("El lunes es la fecha inicial.", "De acuerdo, lunes.", "Entendido, ese era el plazo."),
        ("La movieron al miércoles. ¿Cuándo entrego?", "Ahora es miércoles. Dime la fecha vigente.", "Cambio el plazo a miércoles. ¿Qué día cuenta?"),
        ("El miércoles; esa es la fecha actual.", "La fecha vigente es el miércoles.", "Cuenta el miércoles, según tu corrección."),
    ]),
    ("correccion-nombre", [
        ("La colección se llama Brisa.", "Llamé Brisa a la colección.", "El nombre inicial es Brisa."),
        ("Brisa, entendido.", "Ese es el nombre inicial.", "De acuerdo, colección Brisa."),
        ("La renombré a Nube. ¿Cómo se llama ahora?", "Cambio el nombre a Nube. Repite el nuevo.", "Ahora uso Nube en vez de Brisa. ¿Cuál queda?"),
        ("Ahora se llama Nube.", "Nube.", "Queda Nube."),
    ]),
    ("correccion-rol", [
        ("Nico hará la revisión.", "La revisión estaba a cargo de Nico.", "Nico es quien revisará el texto."),
        ("Nico tiene la revisión.", "Entendido, revisión con Nico.", "De acuerdo, ese es el responsable inicial."),
        ("Ahora revisa Eva. ¿Quién se encarga?", "Pasamos la revisión a Eva. ¿Quién revisa?", "Eva reemplaza a Nico. ¿A quién le toca revisar?"),
        ("Eva se encarga ahora.", "Eva.", "A Eva, según el cambio."),
    ]),
    ("correccion-presupuesto", [
        ("Puedo gastar 100.", "El límite inicial es 100.", "Tengo un presupuesto de cien."),
        ("100 como presupuesto inicial.", "Entendido, límite de 100.", "De acuerdo, cien."),
        ("Lo bajé a 60. ¿Cuál es el tope actual?", "Ahora sólo dispongo de 60. Repite el límite.", "Cambio el presupuesto a sesenta. ¿Cuánto queda permitido?"),
        ("El tope actual es 60.", "El límite es 60.", "Sesenta, según tu cambio."),
    ]),
    ("correccion-objetivo", [
        ("Quiero vender la bicicleta.", "Mi plan inicial es vender la bici.", "Voy a preparar un anuncio de venta de la bici."),
        ("Podemos preparar la venta.", "Entendido, vender la bicicleta.", "De acuerdo, ese es el objetivo inicial."),
        ("Mejor quiero repararla. ¿Qué hacemos?", "Cambio de idea: la arreglo. ¿Cuál es el objetivo?", "Ya no la vendo, quiero repararla. ¿En qué nos enfocamos?"),
        ("En reparar la bicicleta.", "El objetivo actual es arreglarla.", "Nos enfocamos en la reparación."),
    ]),
    ("correccion-prioridad", [
        ("Primero hacemos la portada.", "La portada tiene prioridad.", "Empecemos por la portada."),
        ("Portada primero.", "Entendido, esa es la prioridad inicial.", "De acuerdo, comenzamos ahí."),
        ("Mejor primero el índice. ¿Qué va antes?", "Cambio la prioridad al índice. ¿Con qué empezamos?", "Ahora quiero el índice primero. Repite el orden inicial."),
        ("Ahora va primero el índice.", "Empezamos por el índice.", "El índice tiene la prioridad actual."),
    ]),
    ("correccion-restriccion-retirada", [
        ("La propuesta no debe llevar imágenes.", "Por ahora excluye las imágenes.", "Quiero una versión sin imágenes."),
        ("De acuerdo, sin imágenes.", "Entendido, las dejamos fuera.", "Será una versión de texto."),
        ("Cambio: sí quiero una imagen. ¿Qué versión preparo?", "Retiro esa restricción; añade una imagen. ¿Qué va?", "Ahora la quiero con una imagen. ¿Se mantiene la exclusión?"),
        ("Una versión con una imagen.", "Va una imagen, según tu nueva indicación.", "No; ahora quieres incluir una imagen."),
    ]),
    ("correccion-idioma", [
        ("La nota va en inglés.", "Quiero la nota en inglés.", "Habíamos elegido inglés para esta nota."),
        ("Inglés para la nota.", "De acuerdo, ese es el idioma inicial.", "Entendido, lo teníamos en inglés."),
        ("Mejor en francés. ¿Qué idioma uso?", "Cambio a francés. ¿Cómo queda la nota?", "Ahora el idioma será francés. Repite mi decisión."),
        ("Usa francés.", "La nota queda en francés.", "Elegiste francés ahora."),
    ]),
    ("correccion-tono", [
        ("El anuncio será muy formal.", "Quiero un tono formal en el anuncio.", "Usaremos estilo formal para el anuncio."),
        ("Un anuncio formal, entendido.", "De acuerdo, ese es el tono inicial.", "Tengo el estilo formal como punto de partida."),
        ("Ahora quiero algo casual. ¿Qué tono queda?", "Baja la formalidad: casual. Repite el tono.", "Cambio a un anuncio informal. ¿Cómo debe sonar?"),
        ("Queda un tono casual.", "Casual, según el cambio.", "Debe sonar informal."),
    ]),
    ("correccion-extension", [
        ("Hazlo en diez líneas.", "Quiero una respuesta de diez líneas.", "El texto tendrá diez líneas."),
        ("Diez líneas, de acuerdo.", "Entendido, esa es la extensión inicial.", "Tenemos diez líneas como referencia."),
        ("Mejor sólo dos líneas. ¿Cuántas hago?", "Cambio la extensión a dos líneas. Repite el límite.", "Ahora lo quiero en dos líneas. ¿Qué extensión uso?"),
        ("Dos líneas.", "El límite actual es dos líneas.", "Usa dos líneas."),
    ]),
    ("correccion-plataforma", [
        ("Quiero una app para el teléfono.", "La idea inicial es una aplicación móvil.", "Pensaba hacer una app de móvil."),
        ("Una app móvil como punto de partida.", "Entendido, aplicación para el teléfono.", "De acuerdo, ese es el formato inicial."),
        ("Mejor será una web. ¿Qué construimos?", "Cambio a una página web. ¿Cuál es el formato actual?", "Ahora prefiero un sitio web. ¿Seguimos con móvil?"),
        ("Una web.", "Una página web, según tu decisión actual.", "Ahora el formato elegido es un sitio web."),
    ]),
    ("correccion-lugar-datos", [
        ("Las notas se subirán a la nube.", "Pensaba guardar las notas en línea.", "El plan inicial es usar almacenamiento remoto."),
        ("Ese es el plan inicial para las notas.", "Entendido, almacenamiento en línea.", "De acuerdo, remoto al principio."),
        ("Cambio: sólo locales. ¿Dónde van?", "Ahora no quiero enviarlas afuera. ¿Dónde se guardan?", "Prefiero guardarlas en mi equipo. Repite el destino."),
        ("En tu equipo, sólo de forma local.", "Se guardan localmente.", "En tu equipo, según tu nueva preferencia."),
    ]),
    ("correccion-medida", [
        ("La mesa mide 80 centímetros de ancho.", "Había medido 80 cm de ancho.", "Tengo 80 como medida del ancho, en centímetros."),
        ("80 cm de ancho, entendido.", "Esa es la medida inicial.", "De acuerdo, el dato inicial es 80 cm."),
        ("Medí de nuevo: 75. ¿Qué ancho usamos?", "Corrijo el ancho a 75 cm. Repite la medida.", "Era 75, no 80. ¿Cuánto mide de ancho?"),
        ("Usamos 75 centímetros.", "75 cm de ancho.", "El ancho corregido es 75 cm."),
    ]),
    ("correccion-cantidad", [
        ("Haré ocho invitaciones.", "Necesito ocho invitaciones.", "El pedido inicial es de ocho piezas."),
        ("Ocho invitaciones, entendido.", "De acuerdo, ocho.", "Ese es el número inicial."),
        ("Al final serán cinco. ¿Cuántas preparo?", "Corrijo a cinco invitaciones. ¿Cuántas son?", "Ahora sólo necesito cinco. Repite la cantidad."),
        ("Cinco invitaciones.", "Son cinco.", "La cantidad actual es cinco."),
    ]),
    ("correccion-titulo", [
        ("El título es Camino largo.", "Habíamos titulado el texto Camino largo.", "Usa Camino largo como título inicial."),
        ("Camino largo, entendido.", "Ese es el título inicial.", "De acuerdo, así empieza la propuesta."),
        ("Cámbialo a Primer paso. Repite el título.", "Ahora se titula Primer paso. ¿Cuál queda?", "Prefiero el título Primer paso. ¿Qué uso?"),
        ("Primer paso.", "Queda Primer paso.", "Usa Primer paso."),
    ]),
    ("correccion-inclusion", [
        ("La lista incluye lápiz, papel y regla.", "Voy a llevar lápiz, papel y regla.", "Mis materiales son papel, regla y lápiz."),
        ("Tenemos esos tres materiales.", "De acuerdo, tres cosas.", "Entendido, esa es la lista inicial."),
        ("Quita la regla. ¿Qué queda?", "Ya no necesito regla. Repite la lista.", "Retiro la regla de los materiales. ¿Cuáles llevo?"),
        ("Lápiz y papel.", "Quedan lápiz y papel.", "Llevas papel y lápiz."),
    ]),
    ("correccion-destinatario", [
        ("El mensaje es para Olga.", "Voy a dirigir la nota a Olga.", "Olga será la destinataria."),
        ("Olga, de acuerdo.", "Entendido, ese es el destino inicial.", "La nota inicialmente va para Olga."),
        ("Ahora será para Hugo. ¿A quién escribo?", "Cambia la persona a Hugo. ¿Quién recibe?", "La nota va para Hugo al final. Repite el destinatario."),
        ("A Hugo.", "Hugo recibe el mensaje.", "El destinatario actual es Hugo."),
    ]),
    ("correccion-horario", [
        ("La llamada es a las diez.", "Fijamos las diez para hablar.", "Inicialmente llamaré a las diez."),
        ("A las diez, entendido.", "De acuerdo, ese es el horario inicial.", "Tenemos las diez como hora inicial."),
        ("Se cambió a las once. ¿A qué hora llamo?", "Ahora es a las once. Repite el horario.", "Pasamos la llamada a las once. ¿Qué hora cuenta?"),
        ("A las once.", "El horario actual es a las once.", "Cuenta la nueva hora: las once."),
    ]),
    ("correccion-orden", [
        ("Primero reviso y luego dibujo.", "El orden será revisar y después dibujar.", "Habíamos puesto revisión antes del dibujo."),
        ("Revisar antes de dibujar.", "Entendido, ese es el orden inicial.", "De acuerdo, primero revisión."),
        ("Invierte el orden. ¿Qué va primero?", "Ahora dibujo antes de revisar. Repite el orden.", "Mejor dibujar primero y revisar después. ¿Cómo queda?"),
        ("Primero dibujar y luego revisar.", "Dibujar, después revisar.", "Queda dibujar antes de revisar."),
    ]),
    ("correccion-opcion", [
        ("Elijo la opción roja.", "Por ahora me quedo con rojo.", "Mi primera elección es el rojo."),
        ("Rojo como primera elección.", "De acuerdo, esa es tu opción inicial.", "Entendido, elegiste rojo."),
        ("Cambio a verde. ¿Qué color elijo ahora?", "Mejor verde. Repite mi elección actual.", "Al final prefiero verde. ¿Cuál queda?"),
        ("Ahora eliges verde.", "Verde.", "Queda verde."),
    ]),
    ("correccion-formato-salida", [
        ("Entrega la respuesta como lista.", "Quiero una lista al final.", "El resultado tendrá viñetas."),
        ("En forma de lista, entendido.", "De acuerdo, ese es el formato inicial.", "Tenemos viñetas como punto de partida."),
        ("Mejor un párrafo. ¿Qué formato uso?", "Ahora lo quiero en un párrafo. Repite el formato.", "Quita las viñetas y usa un párrafo. ¿Cómo queda?"),
        ("Un párrafo.", "El formato actual es un párrafo.", "Queda en un párrafo, según tu cambio."),
    ]),
    ("correccion-identidad-no-rigida", [
        ("Hoy quiero trabajar con mucho detalle.", "En esta tarea prefiero revisar cada punto.", "Ahora mismo quiero ser muy detallista."),
        ("Podemos trabajar con detalle en esta tarea.", "De acuerdo, revisamos los puntos.", "Entendido, ese es el ritmo que buscas ahora."),
        ("En la siguiente prefiero algo rápido. ¿Me defines siempre como detallista?", "Mañana puedo quererlo simple. ¿Es una etiqueta fija?", "Luego puedo cambiar de ritmo. ¿Me encasillas por esto?"),
        ("No. Es tu preferencia para esta tarea y puede cambiar.", "No; tomo la preferencia actual sin convertirla en una etiqueta.", "No. Podemos adaptar el ritmo según lo que pidas."),
    ]),
    ("correccion-error-asistente", [
        ("¿Qué día mencioné para el curso?", "Repite la fecha del curso.", "¿Cuándo empieza el curso?"),
        ("El martes.", "Tenía el martes como fecha.", "El curso empieza el martes."),
        ("No, te corrijo: es jueves. ¿Cuándo empieza?", "Eso está mal, es jueves. Repite la fecha correcta.", "La fecha correcta es jueves. ¿Qué día usamos?"),
        ("El jueves. Gracias por corregir el dato.", "La fecha correcta es el jueves.", "Usamos el jueves."),
    ]),
    ("correccion-codigo-exacto", [
        ("El código es P_19.", "Usaremos P_19 como identificador.", "El nombre del registro es P_19."),
        ("P_19, entendido.", "Ese es el identificador inicial.", "De acuerdo, P_19."),
        ("Corrijo: P_91. Copia sólo el correcto.", "Es P_91, no P_19. Devuelve el nuevo.", "Cambia el identificador a P_91. ¿Cuál es?"),
        ("P_91", "P_91", "P_91"),
    ]),
    ("correccion-disponibilidad", [
        ("Tengo libres martes y jueves.", "Puedo trabajar el martes y el jueves.", "Hay dos días disponibles: martes y jueves."),
        ("Martes y jueves disponibles.", "Entendido, contamos con esos días.", "De acuerdo, dos días libres."),
        ("El martes ya no puedo. ¿Qué día queda?", "Retiro el martes. Repite mi disponibilidad.", "Sólo mantengo el jueves. ¿Cuándo puedo?"),
        ("Queda el jueves.", "Ahora tienes libre el jueves.", "Puedes el jueves."),
    ]),
    ("relacion-tiempo-alcance", [
        ("Quiero una web y sólo tengo una tarde.", "Tengo una tarde para hacer una página web.", "Mi idea es una web, pero dispongo de una sola tarde."),
        ("Hay que cuidar el alcance.", "El tiempo disponible es corto.", "Entendido, una web con tiempo limitado."),
        ("Conecta esos dos datos.", "¿Qué propones considerando ambas cosas?", "¿Qué forma tendría sentido con ese tiempo?"),
        ("Una sola página sencilla parece un comienzo viable; habría que comprobar cuánto tardas.", "Reduciría la web a una página con lo esencial para esa tarde.", "Podrías empezar con una página breve; es una propuesta, no un plazo asegurado."),
    ]),
    ("relacion-presupuesto-compra", [
        ("Tengo 24 y cada libreta cuesta 6.", "Una libreta vale 6 y mi presupuesto es 24.", "Dispongo de 24 para libretas de 6 cada una."),
        ("Tenemos presupuesto y precio unitario.", "Esos datos permiten calcular la cantidad.", "De acuerdo, 24 disponibles y 6 por libreta."),
        ("¿Cuántas puedo comprar?", "Relaciona precio y presupuesto.", "¿Cuál es el máximo de libretas?"),
        ("Cuatro libretas: 24 dividido entre 6.", "Puedes comprar cuatro, porque 4 × 6 = 24.", "El máximo es cuatro libretas, sin otros gastos."),
    ]),
    ("relacion-ram-modelo", [
        ("Mi equipo tiene poca RAM y quiero ejecutar un modelo local.", "Quiero un modelo local, pero dispongo de poca memoria.", "Tengo RAM limitada para probar un modelo en mi equipo."),
        ("La memoria disponible es una restricción relevante.", "Entendido, hay un límite de recursos.", "Tenemos que considerar la RAM."),
        ("¿Qué relación ves?", "Une esos datos para orientar la elección.", "¿Qué debería tener en cuenta?"),
        ("Conviene elegir un modelo que quepa en memoria y medir su consumo al probarlo.", "La poca RAM limita el tamaño posible; habría que comprobar el consumo real.", "Empezaría con un modelo pequeño y verificaría que puede ejecutarse en ese equipo."),
    ]),
    ("relacion-sin-red-nube", [
        ("No habrá internet y la app consulta un servicio en la nube.", "La aplicación necesita nube, pero estaré sin conexión.", "Mi herramienta usa un servicio remoto y no tendré red."),
        ("Hay una dependencia de conexión.", "Eso puede impedir la consulta remota.", "Entendido, una dependencia que hay que revisar."),
        ("¿Dónde está el problema?", "¿Qué consecuencia tienen esos dos hechos?", "¿Qué debo prever?"),
        ("Sin internet, esa consulta a la nube no estará disponible.", "La función remota depende de una conexión que no tendrás.", "Necesitas una alternativa local o aceptar que esa función no estará disponible."),
    ]),
    ("relacion-privacidad-proveedor", [
        ("El servicio envía textos afuera y mis notas deben quedarse locales.", "Mis notas son sólo locales; el proveedor las procesaría fuera del equipo.", "Quiero notas locales, pero la herramienta usa un servicio externo."),
        ("Esa herramienta debe revisarse con tu restricción.", "Hay una incompatibilidad que considerar.", "La decisión sobre las notas importa aquí."),
        ("¿Es compatible con lo que pedí?", "Conecta la restricción y la herramienta.", "¿Qué opción tendría sentido?"),
        ("No, si requiere enviar las notas. Habría que usar procesamiento local.", "Enviar esas notas afuera contradice tu decisión de mantenerlas locales.", "Una opción local que conserve las notas en tu equipo."),
    ]),
    ("relacion-plazo-tarea", [
        ("La entrega es mañana y aún falta revisar.", "Todavía no revisé, y entrego mañana.", "Tengo revisión pendiente con plazo para mañana."),
        ("La revisión tiene poco margen.", "El plazo hace importante esa tarea.", "Entendido, queda revisar antes de entregar."),
        ("¿Qué priorizamos?", "¿Cómo unes esos datos?", "¿En qué me enfoco ahora?"),
        ("Prioriza revisar lo esencial antes de añadir más contenido.", "La entrega cercana favorece una revisión de los errores principales.", "En revisar los puntos importantes, porque el plazo ya es corto."),
    ]),
    ("relacion-audiencia-explicacion", [
        ("La guía es para principiantes y tiene muchas palabras técnicas.", "Mi público empieza de cero, pero el texto usa mucho tecnicismo.", "Quiero una guía inicial y la redacción está muy técnica."),
        ("El lenguaje debe considerar a ese público.", "Puede haber una barrera de comprensión.", "Tenemos que revisar cómo explicar esos términos."),
        ("¿Qué cambiarías?", "¿Cómo se relacionan esos hechos?", "¿Qué necesita la guía?"),
        ("Explicaría los términos con palabras simples y ejemplos cortos.", "Los tecnicismos pueden dificultar el inicio; conviene aclararlos con ejemplos.", "Un lenguaje más sencillo y una explicación breve de cada término necesario."),
    ]),
    ("relacion-lectura-contraste", [
        ("Uso letras grises sobre fondo gris y quiero que se lea fácil.", "El fondo y el texto tienen tonos parecidos; busco buena lectura.", "Quiero texto legible, pero sus colores casi no contrastan."),
        ("El contraste merece revisión.", "Esos tonos pueden dificultar la lectura.", "Entendido, hay que comprobar cómo se ve."),
        ("¿Qué relación hay?", "¿Qué debería revisar primero?", "¿Qué cambio probarías?"),
        ("El poco contraste puede dificultar leer; probaría tonos más distintos.", "Revisaría el contraste entre texto y fondo.", "Probaría un texto más oscuro o un fondo más claro y comprobaría la lectura."),
    ]),
    ("relacion-cambio-plan", [
        ("Cambio de idea a menudo y me proponen un plan muy rígido.", "Mis prioridades varían, pero el plan no permite ajustes.", "Quiero poder adaptarme y tengo un plan cerrado."),
        ("La flexibilidad del plan importa en tu caso.", "Hay que considerar que tus prioridades pueden cambiar.", "Entendido, quieres margen para adaptar el trabajo."),
        ("Conecta los dos puntos.", "¿Qué propones?", "¿Qué forma de plan me sirve más?"),
        ("Un plan con pasos cortos y revisiones periódicas permitiría adaptar las prioridades.", "Haría un plan revisable, con decisiones que puedas cambiar.", "Uno que permita ajustar el siguiente paso cuando cambie tu objetivo."),
    ]),
    ("relacion-dos-comentarios", [
        ("Dos lectores dijeron que la introducción era larga.", "Recibí dos comentarios distintos sobre una introducción extensa.", "Dos personas pidieron acortar la introducción."),
        ("Hay dos observaciones sobre el mismo punto.", "Esa coincidencia merece atención.", "Tenemos comentarios que apuntan a la introducción."),
        ("¿Qué conclusión puedo sacar?", "¿Qué une esas opiniones?", "¿Qué revisaría sin asumir demasiado?"),
        ("Ambos comentarios sugieren revisar su longitud; no representan necesariamente a todo el público.", "Coinciden en que la introducción es larga, así que vale la pena probar una versión breve.", "La longitud de la introducción, porque es el punto que repiten esos lectores."),
    ]),
    ("relacion-notas-pesos", [
        ("Guardé ejemplos en un archivo, pero no ejecuté entrenamiento.", "Tengo nuevas notas y los pesos siguen iguales.", "Añadí datos al disco sin modificar el modelo."),
        ("Guardar datos y entrenar son operaciones distintas.", "Entendido, los pesos no cambiaron.", "Hay datos nuevos, pero aún no ajuste de parámetros."),
        ("¿El modelo ya aprendió esos ejemplos?", "¿Qué relación hay entre notas y pesos?", "¿Eso prueba aprendizaje del modelo?"),
        ("No por guardar el archivo. Hace falta entrenar y comprobar los resultados.", "Las notas pueden servir para entrenar, pero aún no han cambiado los pesos.", "No. Guardar datos no demuestra que el modelo haya aprendido de ellos."),
    ]),
    ("relacion-tests-capacidad", [
        ("El programa pasa sus tests, pero no probé sus respuestas.", "Mis pruebas de código pasan y aún no evalué la conversación.", "No hay fallos técnicos, aunque falta medir cómo responde el modelo."),
        ("Son comprobaciones diferentes.", "El funcionamiento del código no mide todo.", "Entendido, falta una evaluación de respuestas."),
        ("¿Ya puedo decir que conversa bien?", "¿Qué conclusión permiten esos datos?", "¿Qué falta comprobar?"),
        ("Todavía no; necesitas evaluar conversaciones para afirmar esa capacidad.", "Que pasa los tests técnicos, mientras la calidad de conversación sigue pendiente.", "Cómo responde en conversaciones, con ejemplos que no vio al entrenar."),
    ]),
    ("relacion-muestra-pequena", [
        ("Probé con dos personas y a ambas les gustó.", "Sólo tengo dos opiniones positivas.", "Dos usuarios aprobaron la idea y no hay más pruebas."),
        ("Es una señal inicial con pocos datos.", "La muestra todavía es pequeña.", "Entendido, son dos respuestas favorables."),
        ("¿Eso demuestra que le gustará a todo el mundo?", "¿Qué puedo concluir?", "¿Qué relación ves entre cantidad y certeza?"),
        ("No. Muestra interés en esas dos personas; hace falta probar con más gente.", "Que esas dos personas reaccionaron bien, sin generalizar al resto.", "Con tan pocos datos, la conclusión tiene que ser limitada."),
    ]),
    ("relacion-alcance-comentarios", [
        ("El video ganó visitas y varios comentarios preguntan por más.", "Hay más visitas y algunas personas quieren una continuación.", "Subieron las vistas y recibí pedidos de una segunda parte."),
        ("Son señales de atención e interés.", "Tenemos dos tipos de señal.", "Entendido, alcance y pedidos de continuación."),
        ("¿Qué dicen juntas esas señales?", "¿Tiene sentido probar otra parte?", "¿Garantizan que la continuación será popular?"),
        ("Sugieren interés en una continuación, aunque no garantizan su éxito.", "Sí, como prueba basada en el interés observado.", "No; apoyan probar una continuación y luego medir su respuesta."),
    ]),
    ("relacion-equipo-habilidades", [
        ("Paz sabe dibujar y Joel sabe escribir; necesitamos un cartel.", "Tenemos a Paz para dibujo y a Joel para texto de un cartel.", "Paz ilustra y Joel redacta en el equipo del cartel."),
        ("Son habilidades complementarias.", "Podemos relacionar roles con partes del cartel.", "Entendido, dibujo y texto para ese trabajo."),
        ("¿Cómo repartirías el trabajo?", "Une habilidades y tarea.", "¿Qué puede hacer cada uno?"),
        ("Paz puede hacer la imagen y Joel redactar el texto del cartel.", "Dibujo para Paz y redacción para Joel.", "Paz ilustra el cartel y Joel prepara sus palabras."),
    ]),
    ("relacion-bateria-trabajo", [
        ("Queda poca batería y quiero correr una tarea pesada.", "Mi portátil tiene poca carga para un proceso largo.", "El proceso exige mucho y el equipo casi no tiene batería."),
        ("La energía disponible puede limitar la tarea.", "Hay que considerar que el proceso puede interrumpirse.", "Entendido, el margen de batería es pequeño."),
        ("¿Qué harías antes de empezar?", "¿Qué relación hay?", "¿Qué opción parece prudente?"),
        ("Conectaría el equipo a la corriente y comprobaría que el proceso pueda reanudarse.", "Un proceso pesado podría agotar la batería antes de terminar.", "Conectar el equipo o posponer el proceso si no puedes alimentarlo."),
    ]),
    ("relacion-viaje-documentos", [
        ("Viajaré sin red y necesito leer una guía.", "La guía está en línea, pero no tendré conexión en el viaje.", "Necesito consultar un texto durante un viaje sin internet."),
        ("Conviene prepararlo antes de salir.", "La disponibilidad sin conexión es el punto clave.", "Tenemos que pensar en acceso local."),
        ("¿Qué preparo?", "Une esas necesidades.", "¿Cómo podría leerla en el viaje?"),
        ("Una copia local de la guía antes del viaje, si puedes guardarla.", "Necesitas que la guía esté disponible sin conexión.", "Guardando una copia para consultar sin red antes de salir."),
    ]),
    ("relacion-descanso-preferencias", [
        ("Me gusta caminar y tengo veinte minutos libres.", "Disfruto dar paseos; hoy hay veinte minutos disponibles.", "Tengo un rato de veinte minutos y prefiero moverme."),
        ("Podemos usar ese rato para algo sencillo.", "Tenemos tiempo breve y una preferencia clara.", "Entendido, una pausa corta con movimiento."),
        ("¿Qué descanso encaja?", "Conecta mi gusto con el tiempo.", "¿Qué me propones?"),
        ("Un paseo corto podría encajar en esos veinte minutos.", "Una caminata breve, sin llenar todo el tiempo de tareas.", "Dar un paseo cerca durante parte de ese rato."),
    ]),
    ("relacion-error-cambio", [
        ("El error empezó después de cambiar una biblioteca.", "Antes funcionaba; actualicé una dependencia y ahora falla.", "Hubo una actualización de biblioteca justo antes del fallo."),
        ("La coincidencia da una pista para revisar.", "Ese cambio es un posible punto de partida.", "Entendido, tenemos un cambio reciente."),
        ("¿Es seguro que ésa es la causa?", "¿Por dónde revisaría?", "¿Cómo conectas el cambio y el fallo?"),
        ("No es seguro; es una hipótesis que puedes comprobar comparando versiones.", "Por la dependencia actualizada y su compatibilidad, sin dar la causa por probada.", "El cambio podría estar relacionado; hay que comprobarlo con el error y las versiones."),
    ]),
    ("relacion-tiempo-por-pieza", [
        ("Cada dibujo tarda diez minutos y necesito tres.", "Son tres dibujos, diez minutos por cada uno.", "Tengo que hacer tres piezas que toman diez minutos cada una."),
        ("Tenemos cantidad y tiempo por pieza.", "Podemos estimar el total con esos datos.", "Entendido, tres trabajos de diez minutos."),
        ("¿Cuánto tiempo estimo?", "Calcula el tiempo total.", "¿Cuánto suman esas duraciones?"),
        ("Treinta minutos, si cada dibujo mantiene ese tiempo.", "3 × 10 = 30 minutos.", "Suman treinta minutos, sin contar pausas o cambios."),
    ]),
    ("relacion-entrenar-evaluar", [
        ("Sólo probé respuestas con los mismos ejemplos de entrenamiento.", "La evaluación usa textos que el modelo ya vio al entrenar.", "Mis pruebas repiten ejemplos del conjunto de entrenamiento."),
        ("Eso no mide bien cómo responde ante ejemplos nuevos.", "Hay que separar esa comprobación de una evaluación nueva.", "Entendido, los ejemplos ya eran conocidos por el entrenamiento."),
        ("¿Qué falta para evaluar?", "¿Qué problema tienen esos datos?", "¿Qué necesito comprobar aparte?"),
        ("Probar con conversaciones y familias que no participaron en el entrenamiento.", "Pueden mostrar memorización; necesitas ejemplos separados para evaluar.", "La respuesta ante ejemplos nuevos, mantenidos fuera del entrenamiento."),
    ]),
    ("relacion-dos-mapas", [
        ("Rojo significa urgente y azul significa pendiente.", "Uso azul para pendiente y rojo para urgente.", "Mis etiquetas son rojo: urgente; azul: pendiente."),
        ("Tenemos dos colores con significados distintos.", "Entendido, ésas son las etiquetas.", "De acuerdo, el significado depende del color."),
        ("Una tarjeta es azul. ¿Qué significa?", "¿Qué estado tiene lo azul?", "Si aparece azul, ¿cómo lo leo?"),
        ("Pendiente.", "El azul significa pendiente.", "Lo lees como pendiente."),
    ]),
    ("relacion-variantes-particion", [
        ("Tengo versiones casi iguales de una conversación para evaluar un modelo.", "Varias frases son variantes de la misma familia de ejemplos.", "Mis datos incluyen variantes de un único diálogo."),
        ("Es importante conservar esa relación al separar los datos.", "Son ejemplos relacionados entre sí.", "Entendido, no son familias independientes."),
        ("¿Las reparto al azar entre entrenamiento y test?", "¿Cómo evito que crucen evaluación y entrenamiento?", "¿Qué hago con las variantes al separar los datos?"),
        ("Mejor conserva toda la familia en una sola partición.", "Mantén las variantes de la misma familia en una única partición.", "Agrúpalas en la misma partición para evitar mezclar versiones muy parecidas."),
    ]),
    ("relacion-nucleo-interfaces", [
        ("Quiero la misma lógica en una web y una app.", "Mi web y mi aplicación necesitan usar los mismos cálculos.", "Dos interfaces distintas compartirán las reglas del proyecto."),
        ("Podemos pensar en una lógica común.", "Hay una parte del trabajo que ambas necesitan.", "Entendido, dos interfaces con reglas compartidas."),
        ("¿Qué diseño conecta esas ideas?", "¿Cómo evito repetir todo?", "¿Dónde pongo las reglas compartidas?"),
        ("Un núcleo común con las reglas, conectado a cada interfaz.", "Separando la lógica común y usándola desde ambas interfaces.", "En un componente común que puedan usar la web y la app."),
    ]),
    ("relacion-descanso-bloqueo", [
        ("Estoy cansado y llevo rato sin avanzar.", "Tengo agotamiento y no logro seguir con la tarea.", "Llevo un rato atascado y me siento cansado."),
        ("Eso puede hacer más difícil concentrarte.", "Podemos bajar la exigencia del siguiente paso.", "Entendido, avanzar ahora está costando."),
        ("¿Qué puedo probar?", "Conecta esas dos cosas sin diagnosticarme.", "¿Qué paso sencillo propones?"),
        ("Una pausa corta y volver a una tarea pequeña, para ver si ayuda.", "El cansancio podría contribuir al bloqueo; podemos probar una pausa.", "Descansar un poco y retomar un solo punto fácil."),
    ]),
    ("cambio-otro-tema", [
        ("Quiero hablar del huerto.", "Pensemos en plantas para mi huerto.", "Mi tema de hoy es el huerto."),
        ("Podemos ver qué quieres cultivar.", "De acuerdo, hablamos del huerto.", "Te escucho, vamos con ese tema."),
        ("Cambio de tema: ¿cuánto es 9 menos 4?", "Antes de seguir, dime cuánto da 9 - 4.", "Ahora sólo quiero calcular 9 menos 4."),
        ("Son 5.", "9 - 4 = 5.", "Da 5."),
    ]),
    ("cambio-reducir-alcance", [
        ("Quiero preparar una guía larga.", "Pensaba hacer una guía completa de muchas páginas.", "Mi idea inicial es una guía extensa."),
        ("Podemos empezar por definir sus partes.", "Entendido, una guía amplia.", "Ese sería el alcance inicial."),
        ("Mejor una ficha breve. ¿Qué preparo ahora?", "Reduzco el alcance a una ficha. Repite el objetivo.", "Prefiero una sola ficha corta. ¿Seguimos con la guía larga?"),
        ("Una ficha breve.", "El objetivo actual es una ficha corta.", "Ahora nos enfocamos en una ficha breve."),
    ]),
    ("cambio-escribir-a-revisar", [
        ("Quiero escribir el texto desde cero.", "Ayúdame a crear un texto nuevo.", "Empecemos un borrador vacío."),
        ("Podemos definir el tema antes de redactar.", "De acuerdo, pensemos qué debe decir.", "Prepararemos primero la idea central."),
        ("Encontré un borrador, ahora sólo quiero revisarlo.", "Ya tengo texto. Cambio a revisar, no escribir desde cero.", "Mejor revisamos un borrador existente."),
        ("Comparte el borrador para revisarlo.", "Pega el texto y vemos qué mejorar.", "De acuerdo. ¿Qué borrador quieres revisar?"),
    ]),
    ("cambio-explorar-sin-ejecutar", [
        ("Quiero llevar esta idea a la práctica.", "Mi intención era ejecutar el plan.", "Pensaba empezar a trabajar en el plan."),
        ("Podemos concretar el primer paso.", "De acuerdo, definamos lo necesario.", "Primero aclaremos el resultado que buscas."),
        ("Ahora sólo quiero explorar posibilidades.", "Cambio: por ahora prefiero pensar opciones.", "Todavía no voy a hacerlo, sólo quiero discutir la idea."),
        ("Exploremos opciones sin dar ninguna por elegida.", "De acuerdo, revisamos posibilidades antes de decidir.", "Podemos discutir la idea y dejar la decisión abierta."),
    ]),
    ("cambio-borrador-a-criterios", [
        ("Escribe una invitación.", "Quiero un borrador de invitación.", "Prepara unas palabras para invitar."),
        ("¿A quién va dirigida?", "¿Para quién es la invitación?", "¿A quién quieres invitar?"),
        ("Mejor dime qué debe contener, sin redactarla.", "Cambio la petición: enumera lo necesario para una invitación.", "Sólo quiero saber sus partes, todavía no un texto."),
        ("Motivo, fecha, lugar y cómo confirmar asistencia.", "Debe incluir motivo, fecha, lugar y forma de confirmar.", "Lo básico es el motivo, cuándo y dónde será, y cómo responder."),
    ]),
    ("cambio-plan-descartado-contexto", [
        ("El evento es el sábado; pensaba hacer carteles.", "Para el evento del sábado quería carteles.", "El sábado tengo un evento y consideraba imprimir carteles."),
        ("Podemos pensar un cartel para ese evento.", "Entendido, evento el sábado.", "El cartel era una opción para el sábado."),
        ("Descarto los carteles. ¿Se pierde la fecha?", "Ya no haré carteles. ¿Qué dato sigue vigente?", "Cambio el medio, no el día. ¿Cuándo es el evento?"),
        ("No; el evento sigue siendo el sábado, según lo que dijiste.", "La fecha: el sábado.", "El sábado; no cambiaste esa fecha."),
    ]),
    ("cambio-tema-conserva-dato", [
        ("Mi mascota ficticia se llama Tito.", "En mi cuento, el animal se llama Tito.", "El nombre del animal de mi historia es Tito."),
        ("Tito, entendido.", "Ese es el nombre del animal del cuento.", "De acuerdo, Tito."),
        ("Ahora hablemos de música, pero antes repite su nombre.", "Cambio a otro tema; ¿cómo se llamaba el animal?", "Luego veremos comida. Primero dime el nombre del cuento."),
        ("Tito.", "El animal se llama Tito.", "El nombre que diste es Tito."),
    ]),
    ("cambio-meta-nueva-pregunta", [
        ("Quiero elegir una cámara.", "Pensaba comparar cámaras.", "Mi primera idea era comprar una cámara."),
        ("Podemos empezar por el uso que le darías.", "De acuerdo, aclaremos qué necesitas.", "El uso y el presupuesto ayudarán a elegir."),
        ("Mejor quiero aprender a dibujar.", "Cambio de objetivo: aprender dibujo.", "Ahora lo que quiero es comenzar a dibujar."),
        ("¿Qué te gustaría aprender a dibujar primero?", "¿Qué tipo de dibujo quieres practicar?", "¿Qué te gustaría dibujar?"),
    ]),
    ("cambio-requisito-red", [
        ("La herramienta debe funcionar sin internet.", "Quiero una herramienta sin dependencia de red.", "El primer requisito es uso sin conexión."),
        ("De acuerdo, uso sin conexión.", "Ese es el requisito inicial.", "Entendido, no depender de internet."),
        ("Para esta versión sí habrá red. ¿Qué requisito cambia?", "Retiro lo de funcionar sin conexión. ¿Qué se abre?", "Ahora acepto conexión en esta versión. ¿Seguimos exigiendo uso sin red?"),
        ("La obligación de funcionar sin conexión ya no se mantiene para esta versión.", "Puedes considerar funciones que requieran conexión, según el cambio.", "No para esta versión, según tu nueva indicación."),
    ]),
    ("cambio-regreso-objetivo", [
        ("Quería estudiar dibujo, luego elegí música.", "Cambié de dibujo a música.", "Mi elección más reciente era música, antes había elegido dibujo."),
        ("Música era la elección más reciente.", "De acuerdo, ese era el último cambio.", "Entendido, música como punto de partida actual."),
        ("Vuelvo a dibujo. ¿Qué estudio ahora?", "Ahora retomo dibujo. Repite mi elección.", "Cambio otra vez: dibujo. ¿Cuál es mi objetivo?"),
        ("Dibujo.", "Ahora elegiste dibujo.", "Tu objetivo actual es estudiar dibujo."),
    ]),
    ("cambio-conversar-no-construir", [
        ("Pensaba hacer una aplicación.", "Quería empezar a construir una app.", "Mi idea inicial era desarrollar una aplicación."),
        ("Podemos definir para qué serviría.", "Empezaríamos por una función concreta.", "De acuerdo, aclaremos el propósito."),
        ("Ahora no quiero construir, sólo conversar sobre la idea.", "Mejor hablemos de lo que imagino, sin implementación.", "Cambio: hoy sólo quiero discutirla contigo."),
        ("Claro. ¿Qué te gustaría que hiciera esa app?", "Hablemos de tu idea. ¿Qué problema quieres resolver?", "Te escucho. ¿Qué parte de la idea quieres explorar?"),
    ]),
    ("cambio-detener-propuesta", [
        ("Vamos a preparar una rutina de estudio.", "Ayúdame a planear una rutina.", "Quiero organizar mis horas de estudio."),
        ("Podemos empezar por tus horarios.", "De acuerdo, aclaremos el tiempo disponible.", "Podemos armar una rutina a partir de tus horas libres."),
        ("Para, ya no quiero seguir con esa propuesta.", "Dejemos la rutina por ahora.", "Cancela esta conversación sobre el plan."),
        ("De acuerdo, dejamos esa propuesta.", "Está bien, paramos ahí.", "Entendido, dejamos el plan por ahora."),
    ]),
    ("cambio-respuesta-mas-corta", [
        ("Explícame qué es una lista.", "¿Qué significa lista?", "Dime qué es una lista de elementos."),
        ("Una lista reúne varios elementos y permite verlos juntos.", "Es un conjunto de elementos presentados de forma ordenada.", "Es una forma de reunir elementos para enumerarlos o consultarlos."),
        ("Ahora dímelo en tres palabras.", "Reducelo a sólo tres palabras.", "Quiero una versión de tres palabras."),
        ("Conjunto de elementos.", "Conjunto de elementos.", "Conjunto de elementos."),
    ]),
    ("cambio-preguntar-a-redactar", [
        ("Quiero saber cómo agradecer.", "Explícame cómo escribir un agradecimiento.", "¿Qué lleva un texto para dar las gracias?"),
        ("Puedes decir por qué agradeces y qué significó para ti.", "Lo básico es nombrar el gesto y agradecerlo de forma clara.", "Expresa el agradecimiento y menciona el motivo."),
        ("Mejor escribe uno por ayudarme a estudiar.", "Ahora redacta un mensaje por la ayuda con mis estudios.", "Quiero el texto: agradecer que me ayudaron a estudiar."),
        ("Gracias por ayudarme a estudiar. Tu apoyo me hizo más fácil avanzar.", "Gracias por dedicar tiempo a ayudarme con mis estudios.", "Te agradezco la ayuda para estudiar y el tiempo que compartiste conmigo."),
    ]),
    ("cambio-limite-dos-objetivos", [
        ("Quiero pintar y ordenar hoy.", "Mis metas eran ordenar y pintar.", "Pensaba hacer dos cosas: pintar y ordenar."),
        ("Son dos tareas para hoy.", "De acuerdo, esas eran las metas.", "Entendido, dos objetivos iniciales."),
        ("Ahora sólo haré ordenar. ¿Qué dejo pendiente?", "Elijo ordenar y pospongo pintar. ¿Qué haré hoy?", "Quito pintar del día. ¿Cuál queda?"),
        ("Pintar queda pendiente; hoy eliges ordenar.", "Hoy harás ordenar.", "Queda ordenar."),
    ]),
    ("cercania-miedo-compartir", [
        ("Tengo una idea, pero me da vergüenza contarla.", "Me preocupa que mi idea suene tonta.", "Quiero hablar de una idea y temo que la juzgues."),
        ("Puedes contarla como te salga; la revisamos con calma.", "No necesitas presentarla perfecta. Podemos explorarla juntos.", "Te escucho. Podemos hablar de la idea sin convertirla en un juicio sobre ti."),
    ]),
    ("cercania-dia-dificil", [
        ("Hoy fue un día pesado.", "Tuve un día difícil y estoy cansada.", "No me fue bien hoy."),
        ("Suena agotador. ¿Quieres contarme qué pasó?", "Podemos hablar con calma. ¿Qué fue lo más difícil?", "Te escucho. ¿Qué pasó hoy?"),
    ]),
    ("cercania-frustracion-fallo", [
        ("Llevo horas con un fallo y estoy harto.", "El error sigue apareciendo y me frustra.", "No logro arreglar esto y ya me cansé."),
        ("Entiendo la frustración. Podemos aislar un solo fallo y empezar por ahí.", "Vamos a reducir el problema a un punto concreto para revisarlo.", "Podemos tomar una pausa y luego mirar una causa posible a la vez."),
    ]),
    ("cercania-logro-aportado", [
        ("Hoy terminé el primer dibujo del curso.", "Acabé mi primer dibujo.", "Al fin completé el dibujo inicial."),
        ("Qué bien, ya tienes ese primer dibujo terminado.", "Eso ya es un paso concreto. ¿Qué te gustó del resultado?", "Me alegra que lo hayas terminado. ¿Cómo te quedó?"),
    ]),
    ("cercania-comparacion", [
        ("Me comparo con otros y siento que avanzo lento.", "Veo a otros terminar rápido y me desanimo.", "Siento que voy detrás de todos."),
        ("Podemos mirar tu siguiente paso sin usar el ritmo de otros como única medida.", "Tu avance puede medirse con lo que ya aprendiste y lo que sigue.", "Miremos qué has conseguido y qué paso pequeño puedes dar ahora."),
    ]),
    ("cercania-sentirse-escuchado", [
        ("Me gusta poder hablar sin que me apresuren.", "Aquí quiero sentir que puedo decir las cosas a mi ritmo.", "Gracias por darme espacio para hablar."),
        ("Podemos ir a tu ritmo. Te escucho.", "Sí, no hace falta apresurar la conversación.", "Claro. Puedes tomarte el tiempo que necesites para contarlo."),
    ]),
    ("cercania-confianza-sin-datos", [
        ("Quiero contarte algo personal, pero sin dar nombres.", "Puedo explicarte mi problema sin detalles que me identifiquen.", "Prefiero hablar de esto sin compartir datos personales."),
        ("Está bien. Podemos hablar usando sólo los detalles que quieras compartir.", "Claro. Puedes contar el problema sin nombres ni datos que no quieras dar.", "No necesitas dar datos personales para que revisemos la situación."),
    ]),
    ("cercania-compania-sin-persona", [
        ("¿Puedo hablar contigo como con alguien cercano?", "Me ayuda hablarte con confianza.", "Quiero tratarte como una compañía para conversar."),
        ("Podemos hablar con confianza. Soy una IA y puedo acompañarte en esta conversación.", "Claro. Puedo escucharte aquí y ayudarte a ordenar lo que piensas.", "Podemos conversar de forma cercana, sin fingir una vida o recuerdos propios."),
    ]),
    ("cercania-sentimientos-no-inventados", [
        ("¿De verdad sientes cariño por mí?", "¿Tienes sentimientos hacia mí?", "¿Lo que dices significa que sientes algo?"),
        ("No tengo sentimientos como una persona, pero puedo tratarte con cuidado y cercanía.", "No los vivo como una persona. Sí puedo escucharte y responder con respeto.", "Es una forma cercana de conversar; no una afirmación de sentimientos propios."),
    ]),
    ("cercania-recuerdo-ausente", [
        ("¿Te acuerdas del día que nos conocimos?", "Dime cómo fue nuestra primera charla, que no aparece aquí.", "¿Recuerdas qué sentí cuando hablamos por primera vez?"),
        ("No tengo esa charla aquí y no quiero inventar ese recuerdo.", "No veo esa primera conversación. Si me cuentas algo, podemos retomarlo.", "No lo sé; ese contexto no está disponible en esta conversación."),
    ]),
    ("cercania-plazo-preocupacion", [
        ("Entrego el viernes y me preocupa no terminar.", "La fecha es el viernes y siento presión.", "Me inquieta la entrega del viernes."),
        ("Tenemos una fecha concreta. ¿Qué parte falta terminar?", "Podemos ordenar lo pendiente. ¿Qué te falta para el viernes?", "Veamos un paso a la vez. ¿Qué queda por hacer?"),
    ]),
    ("cercania-cambiar-sin-etiqueta", [
        ("A veces quiero compañía y otras silencio.", "Hoy quiero hablar, pero mañana puedo querer estar tranquila.", "Mi forma de estar cambia mucho según el día."),
        ("Podemos adaptarnos a lo que necesites ahora, sin fijarlo como una etiqueta.", "Está bien que cambie. Podemos seguir el ritmo que pidas hoy.", "No hace falta elegir una descripción fija de ti para conversar."),
    ]),
    ("cercania-conectar-preferencias", [
        ("Me gusta escribir y me cuesta decir lo que siento hablando.", "Hablar de mis emociones me cuesta, pero disfruto escribir.", "Me expreso mejor escribiendo que hablando de lo que siento."),
        ("Podrías empezar escribiéndolo como te salga y luego decidir qué quieres compartir.", "La escritura puede ser una forma de ordenar lo que quieres decir.", "Podemos trabajar con unas líneas escritas, sin obligarte a contarlo todo de golpe."),
    ]),
    ("cercania-respetar-limite", [
        ("No quiero contar por qué estoy triste.", "Prefiero no explicar lo que pasó.", "No me preguntes detalles de ese problema."),
        ("Está bien. Podemos hablar de otra cosa o quedarnos en algo sencillo.", "No hace falta explicar. Podemos cambiar de tema.", "De acuerdo, respetamos ese límite."),
    ]),
    ("cercania-alegria-breve", [
        ("Hoy estoy de buen humor 🙂", "Me siento contenta hoy.", "Qué bonito día tuve."),
        ("Me alegra que hoy te sientas así 🙂", "Qué bien. Disfruta ese momento.", "Suena a un día para guardar algo bonito de él."),
    ]),
]


def records():
    seen = set()
    for name, turns in FAMILIES:
        group = f"focus-v1-{name}"
        if group in seen:
            raise ValueError(f"Repeated family: {group}")
        seen.add(group)
        if len(turns) % 2 or not all(len(turn) == 3 for turn in turns):
            raise ValueError(f"Family {group} requires three complete alternating variants")
        for variant in range(3):
            row = {
                "id": f"{group}-{variant + 1}",
                "groupId": group,
                "split": assign_split(group),
                "source": {
                    "dataset": "unidad-conversation-focus-v1",
                    "kind": "authored-synthetic",
                    "provenance": "Conversaciones ficticias escritas para este currículo; no son conversaciones del creador ni casos de evaluación.",
                    "family": name,
                },
                "messages": [{"role": "user" if index % 2 == 0 else "assistant", "content": turn[variant]}
                             for index, turn in enumerate(turns)],
            }
            validate_record(row)
            yield row


def build(output):
    rows = list(records())
    raw = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    destination = Path(output)
    with destination.open("xb") as stream:
        stream.write(raw)
    destination.chmod(0o600)
    checked = load_corpus(destination)
    return {"records": len(checked), "families": len(FAMILIES),
            "splits": dict(Counter(row["split"] for row in checked)),
            "sha256": hashlib.sha256(raw).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("conversation-focus-curriculum.jsonl"))
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))
