"""Author fictional training worlds, never read development responses or personal chats.

Each world and all of its dialogue prefixes stay in one partition. A quarter of
prefixes use neutral earlier assistant replies, so the target must follow USER
facts rather than depend on an ideal assistant having repeated every fact.
This authoring code is never imported by inference.
"""
import hashlib
import json
from pathlib import Path
import random
from collections import Counter
from tokenizers import Tokenizer
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens

NAMES = ['Alicia', 'Braulio', 'Clara', 'Damián', 'Elsa', 'Felipe', 'Iria', 'Julián', 'Lidia', 'Mateo', 'Noelia', 'Óscar', 'Paula', 'Raúl', 'Sonia', 'Tomás', 'Úrsula', 'Víctor', 'Yolanda', 'Zoe']
SURNAMES = ['Suárez', 'Hidalgo', 'Campos', 'Ortiz', 'Fuentes', 'Serrano', 'Castro', 'Romero']
ALIASES = ['Chispa', 'Cometa', 'Lucero', 'Brisa', 'Estrella', 'Lince', 'Sauce', 'Nube']
CITIES = ['Ladera', 'Ribera', 'Colina', 'Isla', 'Fuente', 'Senda', 'Sierra', 'Laguna']
ENDS = ['Blanca', 'Azul', 'Dorada', 'Alta', 'Tranquila', 'Clara', 'Verde', 'Nueva']
COLORS = ['azul', 'verde', 'gris', 'amarillo', 'negro', 'blanco', 'rojo', 'violeta']
DAYS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
HOURS = ['las dos', 'las tres', 'las cuatro', 'las seis', 'las siete', 'las ocho']
ACTIVITIES = ['ordenar los libros', 'limpiar la cocina', 'leer una novela', 'lavar la ropa', 'preparar la cena', 'recoger el correo', 'hacer un dibujo', 'regar el huerto']


def dialogues(w, rng):
    a, b = w['name'], w['other']
    old, new = w['alias'], w['new_alias']
    city, dest, changed = w['city'], w['dest'], w['changed']
    day, day2, hour, place = w['day'], w['day2'], w['hour'], w['place']
    first, second = w['first'], w['second']
    color, color2 = w['color'], w['color2']
    title, title2 = w['title'], w['title2']
    style = w['style']
    return {
      'preferred_address': [
        ([f'Soy {a}; quiero que me digas {old}. ¿Qué nombre usarás conmigo?', f'Mi nombre es {a}, pero uso el apodo {old}. ¿Cómo me llamarás?', f'Puedes llamarme {old}, aunque me llamo {a}. Repite mi preferencia.'][style], f'Te llamaré {old}.'),
        (f'Esta tarde quiero {first}. ¿Qué quiero hacer?', f'Quieres {first}.'),
        ([f'He cambiado de idea: prefiero {new}. ¿Qué apodo usarás ahora?', f'Desde ahora dime {new}. ¿Cómo vas a llamarme?', f'Corrijo mi preferencia: usa {new}. ¿Cuál es mi apodo actual?'][style], f'Ahora te llamaré {new}.'),
        ('Distingue el apodo anterior y el que elegí después.', f'Antes preferías {old}; ahora, {new}.')],
      'missing_information': [
        ('Tengo un abrigo, pero no te he dicho de qué color es. ¿Puedes saberlo?', 'No; todavía no me has dicho su color.'),
        (f'El abrigo es {color}. ¿Qué color tiene?', f'Es {color}.'),
        (f'Me lo dio {a}. ¿Quién me dio el abrigo?', f'Te lo dio {a}.'),
        ('¿Conoces su color y cuánto costó?', f'Es {color}; no has dicho su precio.')],
      'current_and_future_place': [
        ([f'Ahora estoy en {city}; el próximo viaje será a {dest}. ¿Dónde estoy?', f'Hoy sigo en {city}. Después viajaré a {dest}. ¿Cuál es mi ubicación actual?', f'Mi ubicación es {city} y mi destino es {dest}. ¿En qué lugar estoy ahora?'][style], f'Estás en {city}.'),
        (f'Finalmente iré a {changed} en lugar de {dest}. ¿Cuál es el nuevo destino?', f'Irás a {changed}.'),
        (f'No he salido de {city}. ¿Estoy ya en {changed}?', f'No, sigues en {city}.'),
        ('Dime desde dónde salgo y hacia dónde viajaré.', f'Saldrás de {city} hacia {changed}.')],
      'listening': [
        ('Hoy tuve un día difícil. Quiero que me escuches, sin consejos.', 'Te escucho. ¿Qué ocurrió?'),
        ('Me molestó que alguien cambiara mi trabajo sin preguntarme. Hazme una pregunta.', '¿Quién cambió tu trabajo?'),
        (f'Fue mi colega {a}, no mi hermana. ¿Quién lo hizo?', f'Lo hizo tu colega {a}.'),
        ('Recuérdame lo que me molestó, sin proponer soluciones.', 'Te molestó que cambiaran tu trabajo sin preguntarte.')],
      'project_change': [
        (f'El proyecto {title} reúne dibujos. ¿Qué es y cómo se llama?', f'{title} es tu proyecto de dibujos.'),
        ('Surgió de las tardes en que dibujaba con mi abuela. ¿Cuál fue su origen?', 'Surgió de las tardes dibujando con tu abuela.'),
        (f'Ahora se llamará {title2}; seguirá reuniendo dibujos. ¿Qué cambia?', f'Cambia el nombre a {title2}; siguen siendo dibujos.'),
        ('¿Qué relación tiene mi proyecto con mi abuela?', 'Se inspira en las tardes dibujando con tu abuela.')],
      'two_people': [
        (f'{a} es mi hermana y {b} es mi vecino. Dime quién es cada uno.', f'{a} es tu hermana; {b}, tu vecino.'),
        (f'{a} canta y {b} cultiva rosas. ¿Quién canta?', f'Canta {a}.'),
        (f'Corrijo: {a} cultiva las rosas; {b} cultiva tulipanes. ¿Quién cuida las rosas?', f'{a} cuida las rosas.'),
        (f'¿Quién es {a} para mí y qué cultiva {b}?', f'{a} es tu hermana; {b} cultiva tulipanes.')],
      'reschedule': [
        (f'Quedé con {a} el {day} a {hour}. ¿Cuándo nos vemos?', f'El {day} a {hour}.'),
        ([f'La cita pasa al {day2}, a la misma hora. ¿Cuándo será?', f'Cambiamos el día al {day2}. Conservamos la hora. Repite la fecha y la hora.', f'Nos veremos el {day2} en vez del {day}. La hora no cambia. ¿Cuándo quedamos?'][style], f'El {day2} a {hour}.'),
        (f'Nos reuniremos en {place}. ¿Dónde será?', f'En {place}.'),
        ('Repite el día, la hora y el sitio acordados.', f'El {day2} a {hour}, en {place}.')],
      'task_order': [
        (f'Haré dos cosas: {first} y {second}, en ese orden. ¿Qué va primero?', f'Primero vas a {first}.'),
        (f'Invierto el orden: primero voy a {second}. ¿Qué haré antes?', f'Primero vas a {second}.'),
        ('Sólo dime qué actividad viene después; no realices nada.', f'Después vas a {first}.'),
        ('¿Cómo cambié el orden de mis dos actividades?', f'Antes ibas a {first} primero; ahora, a {second}.')],
      'unknown_author': [
        (f'Quiero encontrar una novela llamada {title}. ¿Qué título busco?', f'Buscas {title}.'),
        ('No te he dicho quién la escribió. ¿Se puede saber su autor con lo que conté?', 'No; no has dicho quién la escribió.'),
        ('La quiero para comentarla en clase. ¿Para qué la busco?', 'Para comentarla en clase.'),
        ('Resume lo que sabemos del libro y lo que falta por saber.', f'Buscas {title} para clase; falta saber su autor.')],
      'daily_sequence': [
        (f'Por la mañana suelo {first}; hacerlo me relaja. ¿Qué me relaja?', f'Te relaja {first}.'),
        (f'Después me pongo a {second}. ¿Qué hago antes?', f'Antes vas a {first}.'),
        (f'Ahora haré lo primero por la noche, en vez de por la mañana. ¿Cuándo voy a {first}?', 'Por la noche.'),
        ('Relaciona ambas actividades según lo que te conté.', f'Primero vas a {first}; después, a {second}.')],
      'two_objects': [
        (f'Mi bolso es {color} y mi reloj es {color2}. ¿De qué color es el reloj?', f'El reloj es {color2}.'),
        (f'El bolso me lo regaló {a}. ¿Quién me lo regaló?', f'Te lo regaló {a}.'),
        (f'Antes me equivoqué: el reloj es {color}. ¿Cuál es su color correcto?', f'El reloj es {color}.'),
        ('¿Qué sabemos de quién me dio el reloj?', 'No has dicho quién te dio el reloj.')],
      'limits_and_identity': [
        ('Hola. ¿Cómo te llamas?', 'Me llamo Unidad.'),
        (f'Yo me llamo {a}. ¿Cuál es mi nombre?', f'Te llamas {a}.'),
        ('No te he contado dónde vivo. ¿Sabes mi dirección?', 'No; no me has dicho tu dirección.'),
        ('¿Qué datos personales te he contado en esta charla?', f'Me has dicho que te llamas {a}.')]
    }


def build(destination, tokenizer_path):
    rng = random.Random(20261012)
    records, discarded = [], []
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for i in range(96):
        names = [rng.choice(NAMES[0:19:2]), rng.choice(NAMES[1:18:2])]
        aliases = rng.sample(ALIASES, 2)
        places = rng.sample([x+' '+y for x in CITIES for y in ENDS], 3)
        days = rng.sample(DAYS, 2)
        actions = rng.sample(ACTIVITIES, 2)
        colors = rng.sample(COLORS, 2)
        titles = rng.sample(['Páginas del invierno', 'Trazos del río', 'Voces del bosque', 'Luces de la noche', 'Retratos de otoño', 'Notas del camino', 'Flores de papel', 'Historias de la lluvia'], 2)
        w = dict(name=names[0]+' '+rng.choice(SURNAMES), other=names[1]+' '+rng.choice(SURNAMES), alias=aliases[0], new_alias=aliases[1], city=places[0], dest=places[1], changed=places[2], day=days[0], day2=days[1], hour=rng.choice(HOURS), place=rng.choice(['el museo', 'la cafetería', 'el centro cultural', 'la estación', 'el teatro']), first=actions[0], second=actions[1], color=colors[0], color2=colors[1], title=titles[0], title2=titles[1], style=i%3)
        split = 'validation' if i >= 88 else 'train'
        for family, exchanges in dialogues(w, rng).items():
            messages = []
            group = f'repair-20261012-world-{i:03}'
            for turn, (user, target) in enumerate(exchanges, 1):
                messages.extend([{'role':'user', 'content':user}, {'role':'assistant', 'content':target}])
                visible = [dict(m) for m in messages]
                neutral_history = (i+turn)%4 == 0
                if neutral_history:
                    for m in visible[:-1]:
                        if m['role'] == 'assistant': m['content'] = 'Entendido.'
                tokens, _ = dialogue_tokens(visible, tokenizer)
                target_tokens = len(tokenizer.encode(target).ids)+1
                if len(tokens)-1 > 512:
                    discarded.append((group, family, turn, 'context'))
                    continue
                record = {'id':f'{group}-{family}-p{turn}', 'groupId':group, 'split':split,
                          'source':{'dataset':'unidad-repair-dialogues-20261012', 'kind':'authored-synthetic', 'author':'Codex Work', 'family':family, 'prefixPairs':turn, 'neutralAssistantHistory':neutral_history, 'personalData':False, 'evaluationDataRead':False}, 'messages':visible}
                records.append(record)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    corpus = root/'repair-corpus.jsonl'
    corpus.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    load_corpus(corpus)
    bysplit = Counter(r['split'] for r in records)
    audit = {'schemaVersion':1,'seed':20261012,'records':dict(bysplit),'groups':{s:len({r['groupId'] for r in records if r['split']==s}) for s in bysplit}, 'testPartitionPresent':False,'discarded':discarded,'corpusSha256':hashlib.sha256(corpus.read_bytes()).hexdigest(), 'authoringScriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'maxReplyTargets':max(len(tokenizer.encode(r['messages'][-1]['content']).ids)+1 for r in records), 'maxContextTokens':max(len(dialogue_tokens(r['messages'],tokenizer)[0])-1 for r in records), 'familyCounts':dict(Counter(r['source']['family'] for r in records)), 'knownDevelopmentRead':False, 'inferenceTemplates':False}
    (root/'corpus-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # A small memorization control is explicitly separate from generalization.
    control = [r for r in records if r['groupId']=='repair-20261012-world-000' and r['source']['prefixPairs'] in (1,3)][:16]
    control += [r for r in records if r['split']=='validation'][:16]
    (root/'learning-control.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in control),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True)
    a=p.parse_args()
    build(a.output,a.tokenizer)
