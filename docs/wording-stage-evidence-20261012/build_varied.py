"""Fictional, development-informed language augmentation. Never used at runtime.

New worlds and held-out wording banks are declared before training. Existing
corpora and partitions remain untouched. Last-reply labels are authored, not
model-generated answers or text copied from development conversations.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens

ROOT=Path(__file__).resolve().parent
RNG=random.Random(2026101202)
NAMES=['Alicia','Beatriz','Diana','Elena','Gabriela','Irene','Marina','Nuria','Olivia','Patricia','Rocío','Teresa']
OTHERS=['Andrés','Bruno','César','Daniel','Esteban','Gabriel','Iván','Jaime','Lucas','Nicolás','Pedro','Rubén']
PLACES=['Alameda','Bahía','Campiña','Estación','Fuente','Laguna','Loma','Ribera','Senda','Villa']
SUFFIXES=['Azul','Clara','del Roble','Dorada','del Pinar','del Viento','Serena','Vieja']
COLORS=['azul','blanco','rojo','verde','gris','negro','violeta','amarillo','naranja','marrón']
DAYS=['lunes','martes','miércoles','jueves','viernes','sábado','domingo']
HOURS=['las dos','las tres','las cuatro','las cinco','las seis','las siete','las ocho','las nueve']
ACTIONS=['ordenar la mesa','limpiar el balcón','leer una revista','preparar la mochila','lavar los platos','pasear por el parque','escuchar una canción','revisar un dibujo','cocinar la cena','recoger la ropa','escribir una carta','regar el jardín']
OBJECTS=['abrigo','bolso','cuaderno','estuche','reloj','sombrero','vaso','cinturón']
CONTENTS=['dibujos','poemas','fotografías','recuerdos','relatos','canciones']

def nonce(rng):
    return ''.join(rng.choice(['ba','ce','da','fi','ga','lu','me','na','pi','ro','sa','te','va','xo']) for _ in range(3)).capitalize()

def world(i):
    rng=random.Random(7000+i)
    cities=rng.sample([p+' '+s for p in PLACES for s in SUFFIXES],3)
    day,day2=rng.sample(DAYS,2)
    a,b=rng.sample(ACTIONS,2)
    title,title2=rng.sample(['Caminos de papel','Estrellas del puerto','Memorias del viento','Sombras de abril','Jardines del silencio','Cuadros de arena','Cartas del invierno','Rutas de cristal'],2)
    return dict(name=rng.choice(NAMES),other=rng.choice(OTHERS),alias=nonce(rng),alias2=nonce(rng),
                city=cities[0],dest=cities[1],dest2=cities[2],day=day,day2=day2,hour=rng.choice(HOURS),
                action=a,action2=b,title=title,title2=title2,color=rng.choice(COLORS),
                obj=rng.choice(OBJECTS),contents=rng.choice(CONTENTS),
                venue=rng.choice(['el teatro','la escuela','el museo','la plaza','el centro cultural']),
                instrument=rng.choice(['el piano','la guitarra','la flauta','el saxofón']),
                plants=rng.choice(['rosas','tulipanes','orquídeas','helechos']),
                plants2=rng.choice(['margaritas','lirios','girasoles','violetas']))

def families(w):
    n,b,z,z2=w['name'],w['other'],w['alias'],w['alias2']
    c,d,d2=w['city'],w['dest'],w['dest2']
    day,day2,h,v=w['day'],w['day2'],w['hour'],w['venue']
    a,a2=w['action'],w['action2']; t,t2=w['title'],w['title2']
    o,col,things=w['obj'],w['color'],w['contents']; ins,plants,plants2=w['instrument'],w['plants'],w['plants2']
    # Each fact and question list has three TRAIN forms and one reserved form.
    # New values alone never count as held-out wording.
    return {
      'address':[
        ([f'Soy {n}, aunque me gusta el apodo {z}.',f'Para hablarme usa {z}; mi nombre completo empieza por {n}.',f'Me conocen como {n}, pero quiero que aquí me digas {z}.',f'{z} es el nombre con el que quiero que te dirijas a mí; me llamo {n}.'],
         ['¿Qué apodo usarás?','Dime cómo me llamarás.','¿Con qué nombre te dirigirás a mí?','Recuérdame la forma de llamarme que elegí.'],f'Te llamaré {z}.'),
        ([f'Quiero {a} esta tarde.',f'Esta tarde me apetece {a}.',f'Mi plan de hoy es {a}.',f'Para hoy tengo pensado {a}.'],
         ['¿Qué quiero hacer?','Repite mi plan.','¿Qué actividad elegí?','¿Qué te he dicho que me apetece?'],f'Quieres {a}.'),
        ([f'Cambié de opinión: prefiero que me llames {z2}.',f'Ya no uses {z}; ahora quiero {z2}.',f'A partir de ahora, mi apodo será {z2}.',f'He elegido otro apodo, {z2}, para reemplazar el anterior.'],
         ['¿Cómo me llamarás ahora?','Dime mi apodo actual.','¿Qué nombre debo oír de ti desde ahora?','¿Con qué apodo me quedé finalmente?'],f'Ahora te llamaré {z2}.'),
        (['','','',''],['Compara el apodo del principio y el de ahora.','¿Cuál era mi apodo y cuál es el nuevo?','Dime mi preferencia anterior y la actual.','Recuérdame cómo cambió la forma de llamarme.'],f'Antes preferías {z}; ahora, {z2}.')],
      'attribute':[
        ([f'Tengo un {o}, pero todavía no te dije su color.',f'No te he contado de qué color es mi {o}.',f'El color de mi {o} no lo he mencionado.',f'Hay un dato que aún no compartí sobre mi {o}: su color.'],
         ['¿Sabes de qué color es?','¿Puedes decir su color con estos datos?','¿Ya conoces ese color?','¿Hay información suficiente para identificar el color?'],'No me has dicho su color.'),
        ([f'Mi {o} es {col}.',f'El color de ese {o} es {col}.',f'Ahora sí te digo el color: {col}.',f'Para completar el dato que faltaba, el {o} resulta ser {col}.'],
         ['¿De qué color es mi objeto?','Recuérdame su color.','¿Cuál es el color que te acabo de decir?','¿Qué color sabemos que tiene?'],f'Es {col}.'),
        ([f'Me lo regaló {n}.',f'{n} fue quien me dio el {o}.',f'Fue un regalo de {n}.',f'Quien me entregó ese regalo fue {n}.'],
         ['¿Quién me lo dio?','Dime de quién fue el regalo.','¿Qué persona me regaló el objeto?','Recuérdame quién hizo ese regalo.'],f'Te lo regaló {n}.'),
        (['','','',''],['¿Conoces el color y cuánto costó?','Dime qué sabes del color y del precio.','¿Sabemos su precio además de su color?','Separa lo que sabes del color y lo que sabes del coste.'],f'Es {col}; no me has dicho su precio.')],
      'location':[
        ([f'Me encuentro en {c} y planeo visitar {d}.',f'Voy a viajar a {d}, pero por ahora estoy en {c}.',f'Mi lugar actual es {c}; el viaje previsto es a {d}.',f'{d} será el lugar al que viaje; mientras tanto permanezco en {c}.'],
         ['¿Dónde estoy ahora mismo?','Dime mi ubicación actual.','¿En qué lugar sigo?','¿Cuál es el lugar en el que me encuentro todavía?'],f'Estás en {c}.'),
        ([f'El destino cambia a {d2}; ya no voy a {d}.',f'Al final viajaré a {d2} en vez de a {d}.',f'Sustituyo el viaje a {d} por uno a {d2}.',f'He descartado {d} y elegido {d2} para el viaje.'],
         ['¿Adónde viajaré al final?','Dime cuál es mi nuevo destino.','¿Qué lugar visitaré ahora?','¿Cuál quedó como destino después del cambio?'],f'Irás a {d2}.'),
        ([f'Aún no salí de {c}.',f'Todavía estoy en {c}.',f'Sigo en {c}; el viaje no ha comenzado.',f'Me mantengo en {c}, sin haber partido.'],
         [f'¿Ya llegué a {d2}?',f'¿Estoy ya en {d2}?',f'¿Puedes decir que ya estoy en {d2}?',f'¿Sería correcto ubicarme ahora en {d2}?'],f'No, sigues en {c}.'),
        (['','','',''],['Dime el lugar de salida y el destino elegido.','¿Dónde sigo y adónde iré?','Repite desde dónde viajo y hacia qué lugar.','Relaciona mi ubicación de ahora con el viaje que finalmente haré.'],f'Estás en {c} y viajarás a {d2}.')],
      'project':[
        ([f'Mi proyecto {t} es una colección de {things}.',f'Estoy reuniendo {things} en un proyecto llamado {t}.',f'{t} es el título de mi colección de {things}.',f'Le puse el nombre {t} al proyecto donde reúno {things}.'],
         ['¿Qué es mi proyecto y cómo se llama?','Dime su nombre y su contenido.','¿Qué reúne y qué nombre tiene?','Recuérdame de qué trata ese proyecto y su título.'],f'Tu proyecto {t} reúne {things}.'),
        (['Nació de lo que compartía con mi abuela.','Las tardes que pasaba con mi abuela inspiraron el proyecto.','Me inspiré en mis recuerdos con mi abuela.','El origen son las experiencias que viví con mi abuela.'],
         ['¿Qué lo inspiró?','Dime de dónde surgió.','¿Cuál fue el origen de la idea?','¿De dónde vino la inspiración para crearlo?'],'Surgió de tus experiencias con tu abuela.'),
        ([f'Cambiaré el título a {t2}, pero conservaré la colección de {things}.',f'Seguirá la colección de {things}; el nombre nuevo será {t2}.',f'Ahora se llama {t2}; su contenido sigue siendo {things}.',f'El proyecto pasa a titularse {t2} sin modificar la colección de {things}.'],
         ['¿Qué cambia y qué se conserva?','Dime el cambio y lo que sigue igual.','¿Se modificó el contenido o el nombre?','Explica qué parte actualicé y cuál mantuve.'],f'El nombre cambia a {t2}; conserva la colección de {things}.'),
        (['','','',''],['¿Qué relación tiene mi proyecto con mi abuela?','Relaciona la colección con mi abuela.','¿Por qué mi abuela importa en esta idea?','¿Cómo se conecta ese proyecto con las experiencias familiares que mencioné?'],'Tu proyecto se inspira en tus experiencias con tu abuela.')],
      'people':[
        ([f'{n} es mi hermana y {b} es mi vecino.',f'Mi hermana se llama {n}; mi vecino, {b}.',f'Tengo una hermana, {n}, y un vecino, {b}.',f'El nombre de mi vecino es {b} y el de mi hermana es {n}.'],
         ['¿Quién es cada persona?','Dime qué relación tengo con cada uno.','¿Quién es mi hermana y quién mi vecino?','Identifica a ambas personas por su relación conmigo.'],f'Tu hermana es {n}; tu vecino es {b}.'),
        ([f'{n} toca {ins}; {b} cuida {plants}.',f'Quien toca {ins} es {n}; {b} se ocupa de {plants}.',f'{b} cultiva {plants} y {n} practica con {ins}.',f'Entre ellos, {n} se dedica a tocar {ins}, mientras {b} cuida {plants}.'],
         [f'¿Quién toca {ins}?','¿Quién es la persona que toca el instrumento?','Dime quién se dedica a la música.','¿A cuál de esas personas corresponde la actividad musical?'],f'Quien toca {ins} es {n}.'),
        ([f'Corrijo: {n} cuida {plants}, mientras {b} cuida {plants2}.',f'Me equivoqué con las plantas: {n} cultiva {plants} y {b} cultiva {plants2}.',f'Ahora {n} cuida {plants}; {b} se ocupa de {plants2}.',f'El dato correcto es que {n} atiende {plants} y {b} atiende {plants2}.'],
         [f'¿Quién cuida {plants}?',f'¿A quién corresponden esos ejemplares de {plants}?',f'Dime quién se ocupa de {plants}.',f'Después de corregirlo, ¿qué persona tiene a su cargo {plants}?'],f'Quien cuida {plants} es {n}.'),
        (['','','',''],[f'¿Quién es {n} para mí y qué cuida {b}?',f'Dime mi relación con {n} y las plantas de {b}.',f'¿Qué parentesco tengo con {n} y qué plantas tiene {b}?',f'Combina dos datos: la relación que tengo con {n} y las plantas a cargo de {b}.'],f'Tu hermana es {n}; {b} cuida {plants2}.')],
      'event':[
        ([f'Tengo un encuentro con {n} el {day} a {h}.',f'Nos veremos {n} y yo el {day}, a {h}.',f'La cita con {n} será el {day} a {h}.',f'He acordado verme con {n}: será a {h} del {day}.'],
         ['¿Qué día y a qué hora es?','Dime cuándo nos veremos.','Repite el día y la hora del encuentro.','¿Para cuándo quedó programada la cita?'],f'El {day} a {h}.'),
        ([f'Pasamos la cita al {day2} sin cambiar la hora.',f'Ahora nos veremos el {day2}; mantenemos la hora acordada.',f'La hora sigue igual, pero el día será {day2}.',f'El {day2} sustituye al día anterior en la cita; no hay cambio de hora.'],
         ['¿Cuándo será ahora?','Dime el nuevo día y la hora.','¿Qué fecha y hora debo recordar?','Recuérdame cuándo quedaremos después de ese cambio.'],f'El {day2} a {h}.'),
        ([f'El encuentro será en {v}.',f'Nos veremos en {v}.',f'El sitio acordado es {v}.',f'Como lugar del encuentro elegimos {v}.'],
         ['¿En qué lugar nos reuniremos?','Dime dónde será la cita.','¿Qué sitio elegimos?','Recuérdame el lugar en el que quedamos.'],f'En {v}.'),
        (['','','',''],['Recuérdame el día, la hora y el sitio.','Dime los tres datos actuales de la cita.','¿Cuándo y dónde será el encuentro?','Reúne el lugar y el momento acordados finalmente.'],f'El {day2} a {h}, en {v}.')],
      'order':[
        ([f'Voy a {a} y después a {a2}.',f'Haré primero esto: {a}. Luego voy a {a2}.',f'Mis actividades son {a} y {a2}; empezaré por {a}.',f'Antes de {a2}, tengo previsto {a}.'],
         ['¿Qué haré primero?','Dime la primera actividad.','¿Por qué tarea empezaré?','¿Cuál de las dos va al comienzo?'],f'Primero vas a {a}.'),
        ([f'Cambio la prioridad: ahora quiero {a2} primero.',f'Invierto las tareas y empezaré por {a2}.',f'Lo primero será {a2}; cambié el orden anterior.',f'He decidido adelantar {a2} y dejar la otra actividad para después.'],
         ['¿Qué haré antes ahora?','Dime la nueva prioridad.','¿Cuál es ahora la primera tarea?','¿Por qué actividad comenzaré con este nuevo orden?'],f'Primero vas a {a2}.'),
        (['Sólo recuerda el orden; no ejecutes ninguna tarea.','No hagas nada por mí: únicamente responde.','Es una pregunta sobre el orden, no una petición para realizarlo.','Quiero recordar la secuencia, sin que realices las actividades.'],
         ['¿Qué va después de la primera actividad?','Dime qué tarea quedó para después.','¿Cuál es la segunda actividad ahora?','¿Qué vendrá a continuación cuando termine lo primero?'],f'Después vas a {a}.'),
        (['','','',''],['¿Qué cambió en el orden?','Compara la prioridad inicial con la actual.','¿Por cuál iba a empezar y por cuál empezaré ahora?','Explica cómo modifiqué la secuencia del comienzo.'],f'Antes ibas a {a} primero; ahora, a {a2}.')],
      'book':[
        ([f'Estoy buscando una novela llamada {t}.',f'Quiero conseguir el libro {t}.',f'El título del libro que busco es {t}.',f'La obra que me interesa encontrar lleva por título {t}.'],
         ['¿Cómo se llama el libro que busco?','Dime el título que mencioné.','¿Qué libro quiero encontrar?','Recuérdame el nombre de esa obra.'],f'Buscas el libro {t}.'),
        (['No he mencionado quién escribió el libro.','Aún no te he dado el nombre de su autor.','La autoría es un dato que no te he contado.','Todavía falta que comparta quién es la persona que escribió esa obra.'],
         ['¿Sabes quién lo escribió?','¿Puedes identificar al autor sólo con lo que conté?','¿Tenemos el dato de la autoría?','¿La conversación permite decir quién escribió el libro?'],'No me has dicho quién lo escribió.'),
        (['Lo quiero para comentarlo en un taller.','Necesito el libro para participar en un taller de lectura.','Voy a comentarlo con otras personas en un taller.','Mi intención es hablar de esa lectura durante un taller.'],
         ['¿Para qué necesito el libro?','Dime para qué lo busco.','¿Qué quiero hacer con esa lectura?','Recuérdame la finalidad por la que quiero conseguirlo.'],'Lo quieres para comentarlo en un taller.'),
        (['','','',''],['Resume qué sabemos de la obra y qué dato falta.','Dime el título, la finalidad y si conocemos al autor.','¿Qué información tengo y cuál sigue sin estar?','Separa los datos conocidos del libro de la información pendiente.'],f'Buscas {t} para un taller; falta saber su autor.')],
      'listening':[
        (['Hoy tuve un día pesado y quiero desahogarme.','Necesito contar algo que me molestó hoy.','Estoy cansado por lo que pasó y quiero hablarlo.','Quisiera compartir contigo una experiencia que me resultó difícil.'],
         ['Escúchame sin darme consejos.','Acompáñame escuchando; no quiero soluciones.','Sólo quiero que me escuches.','Por ahora prefiero ser escuchado, sin recomendaciones.'],'Te escucho. ¿Qué ocurrió?'),
        (['Alguien interrumpió lo que estaba diciendo.','Me molestó que no me dejaran terminar de hablar.','Cuando hablaba, otra persona me cortó.','Lo que sucedió fue que alguien me interrumpió mientras hablaba.'],
         ['Hazme una pregunta sobre lo que pasó.','Pregúntame por eso.','¿Qué me preguntarías para entenderlo?','Ayúdame a contarlo con una pregunta sobre esa situación.'],'¿Quién te interrumpió?'),
        ([f'Fue mi compañero {b}, no mi jefe.',f'Quien me cortó fue {b}, un compañero.',f'Mi compañero {b} fue la persona que lo hizo.',f'La persona responsable fue {b}; trabaja conmigo y no es mi jefe.'],
         ['¿Quién me interrumpió?','Dime quién fue.','¿Qué persona no me dejó terminar?','Identifica a quien me cortó al hablar.'],f'Te interrumpió tu compañero {b}.'),
        (['Sigo sin querer consejos.','Todavía no busco soluciones.','Sólo quiero recordar lo que te conté.','Mi intención continúa siendo contarlo, no recibir recomendaciones.'],
         ['Resume qué situación me molestó.','¿Qué estaba intentando contarte?','Dime qué pasó, sin aconsejarme.','Recuérdame el hecho que me hizo sentir así.'],f'Tu compañero {b} te interrumpió al hablar.')],
      'daily':[
        ([f'Por la mañana suelo {a}; hacerlo me relaja.',f'Me ayuda a relajarme {a} cada mañana.',f'Una cosa que me despeja es {a}; lo hago por la mañana.',f'Cuando quiero relajarme, me sirve {a}, actividad que realizo por la mañana.'],
         ['¿Qué actividad me ayuda a relajarme?','Dime qué me despeja.','¿Qué me sirve para relajarme?','¿Qué actividad relacioné con sentirme más relajado?'],f'Te ayuda {a}.'),
        ([f'Justo después suelo {a2}.',f'Después de esa actividad me pongo a {a2}.',f'Luego hago lo siguiente: {a2}.',f'Al terminar, la actividad que sigue es {a2}.'],
         [f'¿Qué hago antes de {a2}?','¿Qué actividad realizo inmediatamente antes?','Dime cuál de mis actividades ocurre primero.','¿Qué actividad precede a la que acabo de mencionar?'],f'Antes vas a {a}.'),
        (['Ahora lo primero lo haré por la noche, no por la mañana.','Cambié el horario de la primera actividad a la noche.','La primera actividad pasa de la mañana a la noche.','He decidido reservar la noche, en lugar de la mañana, para la primera actividad.'],
         ['¿Cuándo haré ahora la primera actividad?','Dime su nuevo horario.','¿Qué momento elegí finalmente?','¿En qué parte del día quedó esa actividad después del cambio?'],'Ahora será por la noche.'),
        (['','','',''],['Relaciona las dos actividades en su orden.','Dime qué hago primero y qué hago después.','¿Cuál es la secuencia de las dos actividades?','Explica cómo se enlazan ambas cosas según lo que conté.'],f'Primero vas a {a}; después, a {a2}.')]
    }

def build():
    tokenizer=Tokenizer.from_file(str(ROOT.parent/'UNIDAD/models/conversation-reference-c7/tokenizer.json'))
    records=[];lengths=[];target_lengths=[]
    for i in range(136):
        split='train' if i<128 else 'validation'
        w=world(i)
        panels=['varied'] if split=='train' else ['seen_wording','heldout_wording']
        for panel in panels:
            for family,turns in families(w).items():
                history=[]
                for j,(facts,questions,target) in enumerate(turns,1):
                    fi=3 if panel=='heldout_wording' else RNG.randrange(3)
                    qi=3 if panel=='heldout_wording' else RNG.randrange(3)
                    fact,question=facts[fi],questions[qi]
                    user=' '.join(x for x in ([question,fact] if RNG.random()<0.25 else [fact,question]) if x)
                    history.extend([{'role':'user','content':user},{'role':'assistant','content':target}])
                    visible=[dict(m) for m in history]
                    neutral=(i+j)%4==0
                    if neutral:
                        for message in visible[:-1]:
                            if message['role']=='assistant':message['content']='Entendido.'
                    ids,_=dialogue_tokens(visible,tokenizer)
                    assert len(ids)-1<=512
                    nt=len(tokenizer.encode(target).ids)+1
                    assert nt<=128
                    lengths.append(len(ids)-1);target_lengths.append(nt)
                    records.append({'id':f'principles-{i:03}-{panel}-{family}-p{j}',
                                    'groupId':f'principles-world-{i:03}', 'split':split,
                                    'source':{'dataset':'unidad-wording-composition-20261012','kind':'authored-synthetic',
                                              'author':'Codex Work','family':family,'wordingPanel':panel,
                                              'factForm':fi,'questionForm':qi,'prefixPairs':j,
                                              'neutralAssistantHistory':neutral,'personalData':False,
                                              'developmentInformedDesign':True,'runtimeTemplates':False},
                                    'messages':visible})
    # Rehearsal preserves the earlier TRAIN-only templates without moving any
    # old VALIDATION records into training. Selection is fixed, never by loss.
    old=[json.loads(l) for l in (ROOT.parent/'unidad-fix/repair-corpus.jsonl').read_text().splitlines()]
    train=[r for r in old if r['split']=='train']
    records.extend(RNG.sample(train,512))
    corpus=ROOT/'varied-corpus.jsonl'
    corpus.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    load_corpus(corpus)
    # Audit full user strings against already exposed development; all matched
    # records would be rejected rather than silently used as training examples.
    oldroot=ROOT.parent/'unidad-fix'
    protected=set()
    for fn in ['known-dev.json','fresh-probe.json']:
        suite=json.loads((oldroot/fn).read_text())
        protected.update(t['user'] for e in suite['episodes'] for t in e['turns'])
    wp=json.loads((oldroot/'wording-probe.json').read_text())
    protected.update(p['paraphrase'] for p in wp['pairs'])
    overlaps=[{'record':r['id'],'user':m['content']} for r in records if r['split']=='train'
              for m in r['messages'] if m['role']=='user' and m['content'] in protected]
    assert not overlaps, overlaps[:5]
    audit={'seed':2026101202,'records':dict(Counter(r['split'] for r in records)),
           'newTrainingWorlds':128,'newValidationWorlds':8,'rehearsalTrainRecords':512,
           'panels':dict(Counter(r['source'].get('wordingPanel','old_train_rehearsal') for r in records)),
           'heldoutFormIds':[3],'trainFormIds':[0,1,2],
           'maxNewContextTokens':max(lengths),'maxNewReplyTargets':max(target_lengths),
           'sharedTasksAcrossSplits':True,'heldoutTaskFamilies':False,
           'limitation':'Validation tests held-out wording and fictional worlds within practiced task families, not arbitrary open conversation. A separate fresh transfer suite is required.',
           'protectedDevelopmentExactUserOverlaps':overlaps,'priorPartitionsUnchanged':True,
           'sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
           'authoringScriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'varied-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':build()
