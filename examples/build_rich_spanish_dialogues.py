"""Small, explicitly authored fictional dialogues; never an inference path."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ia_propia.conversation.data import assign_split, validate_record


# These examples are written for the project. They are not a person's chat log,
# retrieved answers, or generations from an external service. Each tuple is one
# complete family; every supervised prefix remains in that family's split.
DIALOGUES = (
    (("Hola, quiero organizar mi día.", "Hola. ¿Qué necesitas hacer hoy?"),
     ("Tengo que estudiar y comprar comida.", "Podemos ordenar esas dos tareas. ¿Alguna tiene un horario fijo?"),
     ("La tienda cierra a las seis.", "Entonces conviene comprar antes de las seis y reservar otro rato para estudiar."),
     ("Primero estudiaré una hora.", "De acuerdo. Estudia esa hora y después compra con tiempo antes del cierre.")),
    (("Buenas. ¿Recuerdas cómo me llamo?", "Todavía no me has dicho tu nombre en esta conversación."),
     ("Soy Irene.", "Hola, Irene. ¿Sobre qué te gustaría conversar?"),
     ("Prefiero que me digas Iri.", "De acuerdo, te llamaré Iri."),
     ("¿Cuál es mi nombre y cómo quiero que me llames?", "Tu nombre es Irene y prefieres que te llame Iri.")),
    (("Estoy algo cansado y quiero hablar.", "Te escucho. ¿Cómo ha sido tu día?"),
     ("Tuve varias reuniones y acabé agotado.", "Varias reuniones seguidas pueden resultar agotadoras. Puedes contarme qué fue lo más pesado."),
     ("Ahora no quiero consejos.", "De acuerdo. Te escucho sin darte consejos."),
     ("La última reunión fue muy larga.", "Entiendo. Después de tantas reuniones, esa última tan larga se sumó al cansancio.")),
    (("Voy a preparar una cena para cuatro.", "¿Qué alimentos tienes y hay alguna preferencia que debamos considerar?"),
     ("Tengo arroz, tomates y huevos.", "Con eso podrías preparar arroz con tomate y huevos. ¿Quieres una idea sencilla?"),
     ("Una persona no come huevos.", "Entonces conviene dejar los huevos aparte y servir arroz con tomate para todos."),
     ("Sí, dejaré los huevos aparte.", "Así cada persona puede elegir y la cena respeta esa preferencia.")),
    (("Estoy en Burgos.", "Estás en Burgos. ¿Qué quieres hacer allí?"),
     ("Busco una actividad tranquila.", "Podrías elegir un paseo o una visita tranquila, según lo que te apetezca."),
     ("Me equivoqué: estoy en Soria.", "Gracias por corregirlo. Ahora tomo Soria como tu ubicación."),
     ("¿En qué ciudad estoy?", "Estás en Soria.")),
    (("¿De qué color es mi chaqueta?", "Todavía no me has dicho el color de tu chaqueta."),
     ("La chaqueta es roja y la bufanda es gris.", "Tu chaqueta es roja y tu bufanda es gris."),
     ("La chaqueta ahora es negra.", "De acuerdo: la chaqueta es negra; la bufanda sigue siendo gris."),
     ("¿Qué color tiene cada una?", "La chaqueta es negra y la bufanda es gris.")),
    (("Quiero aprender a tocar la guitarra.", "¿Has tocado antes o empiezas desde cero?"),
     ("Empiezo desde cero y tengo quince minutos al día.", "Podemos empezar con una práctica breve: postura, afinación y un acorde sencillo."),
     ("No tengo guitarra todavía.", "Entonces el primer paso es conseguir una guitarra o acceder a una para practicar."),
     ("¿Por qué cambiaste el primer paso?", "Porque me dijiste que todavía no tienes guitarra. La práctica necesita primero un instrumento.")),
    (("¿Qué diferencia hay entre entrenar una IA y usarla?", "Entrenarla cambia sus parámetros con ejemplos. Usarla aplica los parámetros actuales para producir una respuesta."),
     ("¿Hablarle siempre cambia sus parámetros?", "No necesariamente. Una conversación puede aportar contexto sin realizar entrenamiento."),
     ("Entonces contexto y entrenamiento son distintos.", "Sí. El contexto sirve para la respuesta actual; entrenar modifica los parámetros del modelo."),
     ("Explícalo con pocas palabras.", "Contexto: lo que lee ahora. Entrenamiento: lo que cambia sus parámetros.")),
    (("Tengo una reunión el jueves a las diez.", "La reunión es el jueves a las diez."),
     ("Necesito preparar una presentación.", "Puedes preparar la presentación antes de la reunión del jueves."),
     ("La movieron al viernes a las once.", "Actualizo el dato: la reunión es el viernes a las once."),
     ("¿Qué horario debo tener en cuenta?", "El viernes a las once, que es el horario corregido.")),
    (("Mi hermana se llama Celia y yo soy Bruno.", "Tú eres Bruno y tu hermana se llama Celia."),
     ("Celia vive en Cuenca y yo en Huesca.", "Celia vive en Cuenca y tú vives en Huesca."),
     ("¿Dónde vive mi hermana?", "Tu hermana Celia vive en Cuenca."),
     ("¿Y yo?", "Tú, Bruno, vives en Huesca.")),
    (("Quiero poner nombre a un cuaderno de ideas.", "¿Qué sensación quieres que transmita el nombre?"),
     ("Algo relacionado con caminos.", "Podrías llamarlo Sendero, Cruce o Ruta. Los tres conectan con la idea de caminos."),
     ("Me quedo con Sendero.", "De acuerdo, tu cuaderno de ideas se llamará Sendero."),
     ("¿Qué nombre elegí?", "Elegiste Sendero.")),
    (("No sé por dónde empezar un texto.", "¿Sobre qué quieres escribir y para quién?"),
     ("Sobre un paseo, para un amigo.", "Puedes empezar con una escena del paseo y después contar por qué te acordaste de tu amigo."),
     ("Quiero que sea breve.", "Entonces elige una sola escena y cuéntala en pocas frases."),
     ("¿Qué me sugeriste?", "Elegir una escena del paseo y escribir unas pocas frases para tu amigo.")),
    (("¿Qué significa que una respuesta sea una inferencia?", "Significa que es una conclusión obtenida a partir de pistas, en lugar de un dato confirmado directamente."),
     ("Si digo que estoy mojado, ¿sabes que ha llovido?", "No. La lluvia sería una posibilidad, pero también podrías haberte mojado de otra forma."),
     ("Me caí en una piscina.", "Entonces sabemos por lo que me cuentas que te mojaste al caer en una piscina."),
     ("¿Por qué no debías asegurar que llovió?", "Porque estar mojado no confirma la lluvia. Faltaba conocer la causa.")),
    (("Estoy decidiendo entre leer y salir a caminar.", "¿Qué necesitas más ahora: descanso tranquilo o movimiento?"),
     ("Llevo mucho rato sentado.", "Eso puede hacer que caminar un rato sea una opción útil si te apetece moverte."),
     ("Pero quiero terminar un capítulo.", "Puedes terminar el capítulo y después dar un paseo corto."),
     ("Haré eso.", "De acuerdo: primero terminas el capítulo y luego sales a caminar.")),
    (("Me gustaría cuidar una planta.", "¿Ya tienes una planta o quieres elegir una?"),
     ("Todavía no tengo ninguna.", "Podemos empezar eligiendo una según la luz y el espacio que tengas."),
     ("¿Cada cuánto la riego?", "Necesitamos saber qué planta elegirás. La frecuencia depende de la especie y de sus condiciones."),
     ("Es verdad, primero debo elegirla.", "Sí. Cuando sepamos cuál es, podremos hablar de sus cuidados concretos.")),
    (("Voy a visitar a mi primo el domingo.", "¿Quieres preparar algo para la visita?"),
     ("Le llevaré un libro y una tarta.", "Llevarás un libro y una tarta a tu primo."),
     ("La tarta la llevaré a mi vecina, no a mi primo.", "Gracias por aclararlo: el libro es para tu primo y la tarta para tu vecina."),
     ("¿Qué le llevaré a mi primo?", "Le llevarás un libro.")),
    (("¿Puedes adivinar mi edad?", "No tengo información para saber tu edad."),
     ("Tengo veintiocho años.", "De acuerdo, tienes veintiocho años."),
     ("Era un ejemplo inventado.", "Entonces lo trato como un dato ficticio del ejemplo, no como tu edad real."),
     ("¿Sabes ahora mi edad real?", "No. Me has dado una edad ficticia, pero no tu edad real.")),
    (("Quiero practicar español.", "Podemos conversar en español. ¿Hay algún tema que te interese?"),
     ("Me gusta cocinar.", "Hablemos de cocina. ¿Qué plato disfrutas preparando?"),
     ("Hago sopa cuando hace frío.", "Tiene sentido conectar la sopa con los días fríos. ¿Qué ingredientes sueles usar?"),
     ("Verduras y fideos.", "Preparas sopa de verduras y fideos cuando hace frío.")),
    (("Me preocupa olvidar una tarea.", "¿Qué tarea necesitas recordar?"),
     ("Enviar un documento mañana.", "Puedes anotar que mañana necesitas enviar ese documento."),
     ("¿Ya lo anotaste en mi calendario?", "No he añadido nada a tu calendario en esta conversación."),
     ("Entonces sólo lo hemos hablado.", "Sí. Hemos hablado de la tarea; eso no confirma que esté registrada en un calendario.")),
    (("Quiero explicar qué es una variable.", "En programación, una variable permite asociar un nombre con un valor."),
     ("Pon un ejemplo sencillo.", "Puedes tener una variable llamada edad con el valor 30."),
     ("¿Y si cambio el valor a 31?", "El nombre edad puede mantenerse y su valor pasa a ser 31."),
     ("Relaciónalo con una corrección.", "Al cambiar el valor, usas el dato actualizado. Es parecido a corregir una información anterior.")),
    (("Quiero hablar sobre mi trabajo.", "Te escucho. ¿Qué está pasando en tu trabajo?"),
     ("Estoy pensando en cambiar de proyecto.", "Estás considerando un cambio de proyecto. ¿Qué te lleva a pensarlo?"),
     ("Prefiero que hoy sólo me escuches.", "De acuerdo. Puedes seguir contándome; hoy te escucharé sin proponer soluciones."),
     ("Me cuesta saber qué quiero.", "Parece que todavía estás aclarando lo que quieres. Te escucho.")),
    (("Tengo treinta minutos y dos tareas.", "¿Cuáles son las tareas y cuánto tarda cada una?"),
     ("Una tarda diez minutos y la otra quince.", "Juntas tardan veinticinco minutos, así que entrarían en esos treinta minutos."),
     ("La segunda tarda veinticinco, me equivoqué.", "Con la corrección, juntas tardan treinta y cinco minutos. Necesitarías más tiempo o dejar una para después."),
     ("¿Por qué ya no caben?", "Porque diez más veinticinco son treinta y cinco, cinco minutos más de los que tienes.")),
    (("Soy Nadia y estoy leyendo una novela.", "Hola, Nadia. ¿Qué te está pareciendo la novela?"),
     ("Me interesa cómo cambian los personajes.", "Te interesa la evolución de los personajes. ¿Hay algún cambio que te haya llamado la atención?"),
     ("No quiero contar el final.", "Podemos hablar de los cambios sin revelar el final."),
     ("¿Qué límite puse?", "Pediste no contar el final de la novela.")),
    (("Estoy aprendiendo a dibujar árboles.", "¿Qué parte te resulta más difícil?"),
     ("Las ramas, porque se ven rígidas.", "Quieres que las ramas se vean menos rígidas. Podemos centrarnos en cómo varían sus curvas y direcciones."),
     ("Dame un solo ejercicio.", "Dibuja varias ramas con curvas suaves, cambiando ligeramente la dirección de cada una."),
     ("¿Qué relación tiene con mi dificultad?", "Practicar curvas y direcciones distintas puede ayudarte a evitar que todas las ramas se vean rígidas.")),
    (("¿Qué libro tengo en mi mesa?", "No me has dicho qué libro tienes en tu mesa."),
     ("Hay un libro de astronomía.", "En tu mesa hay un libro de astronomía."),
     ("Ahora lo guardé y dejé uno de historia.", "El libro que dejaste ahora en la mesa es de historia."),
     ("¿Cuál está en la mesa?", "El de historia.")),
    (("Hola. Hoy quiero conversar sin un tema fijo.", "Hola. Podemos ir siguiendo lo que te apetezca contar."),
     ("Primero pensé en música, pero ahora en viajes.", "De acuerdo, sigamos por los viajes."),
     ("En realidad vuelvo a la música.", "Volvemos a la música. ¿Qué te gustaría compartir?"),
     ("¿Por qué cambias de tema conmigo?", "Porque me vas diciendo qué tema prefieres y puedo seguir tus cambios.")),
    (("¿Tienes información sobre mi familia?", "Sólo puedo usar la información que compartas en esta conversación."),
     ("Tengo dos hermanos.", "Me has dicho que tienes dos hermanos."),
     ("Uno se llama Teo.", "Uno de tus hermanos se llama Teo."),
     ("¿Cómo se llama el otro?", "Todavía no me has dicho cómo se llama el otro hermano.")),
    (("Estoy preparando un correo.", "¿A quién va dirigido y qué quieres comunicar?"),
     ("A una compañera, para cambiar la hora de una reunión.", "Puedes indicar la reunión, proponer la nueva hora y preguntarle si le viene bien."),
     ("La nueva hora es a las tres.", "Puedes escribir: ¿Te viene bien cambiar la reunión a las tres?"),
     ("¿Ya enviaste el correo?", "No he enviado ningún correo. Sólo hemos preparado una posible frase.")),
    (("Quiero comprar fruta para tres días.", "¿Cuántas piezas piensas comer cada día?"),
     ("Dos piezas al día.", "Para tres días a dos piezas al día, necesitarías seis piezas."),
     ("Vendrá una amiga y comerá una pieza por día.", "Con una pieza diaria para tu amiga, el total sería nueve piezas para los tres días."),
     ("¿Cómo obtuviste nueve?", "Son seis para ti y tres para tu amiga: seis más tres son nueve.")),
    (("¿Eres Unidad?", "Sí, soy Unidad."),
     ("¿Puedes ayudarme a pensar una idea?", "Podemos explorarla juntos. ¿Qué tienes en mente?"),
     ("Una biblioteca pequeña para mi barrio.", "Podemos pensar cómo conseguir libros, dónde colocarlos y cómo prestarlos."),
     ("Primero quiero hablar de los libros.", "De acuerdo, empecemos por los libros para esa biblioteca.")),
)


def build_records():
    records = []
    for number, dialogue in enumerate(DIALOGUES):
        group = f"unidad-rich-spanish-dialogues-v1-family-{number:03d}"
        messages = []
        for turn, (user, assistant) in enumerate(dialogue, start=1):
            messages.extend(({'role': 'user', 'content': user},
                             {'role': 'assistant', 'content': assistant}))
            record = {'id': f'{group}-prefix-{turn}', 'groupId': group,
                      'split': assign_split(group),
                      'source': {'dataset': 'unidad-rich-spanish-dialogues-v1',
                                 'kind': 'authored-synthetic', 'language': 'es',
                                 'fictional': True, 'prefixPairs': turn},
                      'messages': [dict(message) for message in messages]}
            validate_record(record)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise ValueError('Existing authored corpus is immutable')
    records = build_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
                   for record in records)
    with args.output.open('xb') as stream:
        stream.write(raw)
    manifest = {'purpose': 'training_examples_not_inference_or_evaluation_answers',
                'provenance': 'authored fictional project examples; no owner chat or external generation API',
                'families': len(DIALOGUES), 'records': len(records),
                'splits': dict(Counter(record['split'] for record in records)),
                'sha256': hashlib.sha256(raw).hexdigest(),
                'builderSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'wholeFamilySplits': True, 'supervision': 'final_assistant_reply_per_prefix'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
