"""Authored Spanish templates for explicitly fictional conversation scenes.

Each template contains three complete user/assistant exchanges. Facts come from
the user turns; replies use only that local context. This bank does not generate
data, assign evaluation splits, load a model, or perform any external action.
The curriculum builder owns scene values, family identities and provenance.
"""


TEMPLATES: dict[str, list[dict]] = {
    "greeting": [
        {
            "templateId": "greeting-warm-start",
            "turns": [
                (
                    "Hola, soy {name}. ¿Cómo empezamos esta charla?",
                    "Hola, {name}. Cuéntame qué tienes en mente.",
                ),
                (
                    "Me apetece {activity}, pero antes quiero charlar un poco. ¿Podemos empezar sin un plan?",
                    "Sí. Podemos conversar sin planear nada todavía.",
                ),
                (
                    "Entonces, ¿qué pregunta sencilla me harías sobre {activity}?",
                    "¿Qué te gusta de {activity}?",
                ),
            ],
        },
        {
            "templateId": "greeting-unhurried-chat",
            "turns": [
                (
                    "Buenas, Unidad. Hoy estoy en {place}. ¿Charlamos un rato?",
                    "Buenas. Claro, podemos charlar un rato.",
                ),
                (
                    "Estoy pensando en {activity}. ¿Qué podrías preguntarme sobre eso?",
                    "¿Qué te llamó la atención de {activity}?",
                ),
                (
                    "Iba a dejarlo para mañana, pero he decidido hacerlo hoy. ¿Cuándo lo haré?",
                    "Has decidido hacerlo hoy.",
                ),
            ],
        },
        {
            "templateId": "greeting-a-few-minutes",
            "turns": [
                (
                    "Hola. Tengo {count} minutos libres. ¿Qué charla sencilla podemos tener?",
                    "Podemos hablar de algo que te apetezca ahora.",
                ),
                (
                    "Hoy prefiero {activity} a {other_activity}. ¿Qué opción te he dicho?",
                    "Hoy prefieres {activity}.",
                ),
                (
                    "Antes de irme, ¿qué pregunta corta me harías sobre esa opción?",
                    "¿Qué te apetece de {activity}?",
                ),
            ],
        },
        {
            "templateId": "greeting-name-and-state",
            "turns": [
                (
                    "Hola, me llamo {name}. ¿Cómo puedes llamarme en esta charla?",
                    "Hola, {name}. Puedo llamarte por ese nombre.",
                ),
                (
                    "Vamos a charlar en español. Estoy {feeling}; ¿puedes ir con calma?",
                    "Sí. Podemos conversar con calma.",
                ),
                (
                    "Te he contado mi estado de hoy. ¿Cómo te dije que me siento?",
                    "Hoy te sientes {feeling}.",
                ),
            ],
        },
    ],
    "listening": [
        {
            "templateId": "listening-without-solutions",
            "turns": [
                (
                    "Hoy me siento {feeling} porque {reason}. ¿Puedo contártelo sin que me des soluciones?",
                    "Sí. Te leo; no hace falta buscar una solución ahora.",
                ),
                (
                    "Me cuesta ordenarlo. ¿Podemos ir con calma?",
                    "Claro. Puedes contarlo a tu ritmo.",
                ),
                (
                    "¿Qué motivo te he mencionado? Sólo repítelo sin sacar conclusiones.",
                    "Has mencionado este motivo: {reason}.",
                ),
            ],
        },
        {
            "templateId": "listening-share-as-much-as-needed",
            "turns": [
                (
                    "Quiero hablar de {activity}, no preparar un plan. ¿Puedes escucharme primero?",
                    "Sí. Puedes contarme qué significa para ti.",
                ),
                (
                    "Hoy me siento {feeling}. ¿Hace falta que explique todo?",
                    "No. Puedes compartir sólo lo que te apetezca.",
                ),
                (
                    "Sin recomendaciones, ¿qué pregunta me harías sobre lo que quiero contar?",
                    "¿Qué te gustaría contar de {activity}?",
                ),
            ],
        },
        {
            "templateId": "listening-options-without-a-choice",
            "turns": [
                (
                    "Traigo una idea sobre el {item}, pero necesito hablar antes de decidir. ¿Me escuchas?",
                    "Sí. Puedes explicarla sin tomar una decisión todavía.",
                ),
                (
                    "Las dos opciones tienen los colores {color} y {other_color}. ¿Tenemos que elegir uno ahora?",
                    "No. Podemos hablar de las opciones sin elegir aún.",
                ),
                (
                    "¿Qué dos colores he mencionado? No prefieras ninguno por mí.",
                    "Has mencionado {color} y {other_color}.",
                ),
            ],
        },
        {
            "templateId": "listening-respect-an-unexplained-feeling",
            "turns": [
                (
                    "Estoy {feeling} y quiero conversar un poco. ¿Podemos dejar los consejos para después?",
                    "Sí. Podemos conversar sin entrar en consejos.",
                ),
                (
                    "Todavía no me apetece contar el motivo. ¿Está bien?",
                    "Está bien. No tienes que explicarlo ahora.",
                ),
                (
                    "Sin volver al motivo, ¿qué me preguntarías sobre {activity}?",
                    "¿Qué te interesa de {activity}?",
                ),
            ],
        },
    ],
    "intent_switch": [
        {
            "templateId": "intent-switch-listening-to-a-step",
            "turns": [
                (
                    "Quiero contarte algo de {activity}, sin pasos que cumplir. ¿Puedes escuchar?",
                    "Sí. Cuéntame lo que te apetezca sobre {activity}.",
                ),
                (
                    "Ahora sí quiero empezar. ¿Me propones un primer paso sencillo?",
                    "Dedica unos minutos a {activity}, sin exigirte hacerlo perfecto.",
                ),
                (
                    "¿Qué quiero ahora: seguir hablando o dar ese primer paso?",
                    "Ahora quieres dar un primer paso para {activity}.",
                ),
            ],
        },
        {
            "templateId": "intent-switch-ordering-to-listening",
            "turns": [
                (
                    "Quiero ordenar el {item} y el {other_item}. ¿Puedes proponerme un primer paso?",
                    "Empieza por decidir dónde quieres poner el {item}.",
                ),
                (
                    "No quiero organizar ahora. Prefiero contarte por qué estoy {feeling}. ¿Cambiamos de tema?",
                    "Sí. Podemos dejar el orden para después y escucharte.",
                ),
                (
                    "¿Cuál es mi intención ahora?",
                    "Ahora quieres contar por qué te sientes {feeling}.",
                ),
            ],
        },
        {
            "templateId": "intent-switch-activity-and-day",
            "turns": [
                (
                    "Quiero {activity} el {day}. ¿Puedes repetir ese plan?",
                    "Tu plan es {activity} el {day}.",
                ),
                (
                    "Lo cambié: ahora quiero {other_activity} el {other_day}. ¿Qué plan queda?",
                    "Ahora quieres {other_activity} el {other_day}.",
                ),
                (
                    "No añadas un plan nuevo. ¿Cuál es la actividad actual?",
                    "La actividad actual es {other_activity}.",
                ),
            ],
        },
        {
            "templateId": "intent-switch-a-note-to-a-new-topic",
            "turns": [
                (
                    "Quiero escribir una nota para {name}. ¿Quién será el destinatario?",
                    "La nota será para {name}.",
                ),
                (
                    "He cambiado de idea. No voy a redactarla todavía; sólo quiero hablar de {activity}. ¿Podemos hacer eso?",
                    "Sí. Dejamos la nota y conversamos sobre {activity}.",
                ),
                (
                    "Sin volver a la nota, ¿qué pregunta me harías del tema actual?",
                    "¿Qué te gustaría contar sobre {activity}?",
                ),
            ],
        },
    ],
    "current_question": [
        {
            "templateId": "current-question-two-locations",
            "turns": [
                (
                    "En esta escena, el {item} está en {place} y el {other_item} en {other_place}. ¿Puedes confirmar esos dos datos?",
                    "El {item} está en {place} y el {other_item} en {other_place}.",
                ),
                (
                    "Quiero saber dónde está el {item}, sin hablar del otro. ¿Dónde está?",
                    "El {item} está en {place}.",
                ),
                (
                    "Ahora pregunto por el {other_item}. ¿En qué ciudad está?",
                    "El {other_item} está en {other_place}.",
                ),
            ],
        },
        {
            "templateId": "current-question-a-number-or-a-day",
            "turns": [
                (
                    "En el relato, el {day} leí {count} páginas. ¿Qué datos te estoy dando?",
                    "Que leíste {count} páginas el {day}.",
                ),
                (
                    "Ahora quiero el número, no el día. ¿Cuántas páginas fueron?",
                    "Fueron {count} páginas.",
                ),
                (
                    "Ya tengo el número. ¿En qué día ocurrió?",
                    "Ocurrió el {day}.",
                ),
            ],
        },
        {
            "templateId": "current-question-two-colors",
            "turns": [
                (
                    "Mi {item} es {color} y mi {other_item} es {other_color}. ¿Puedes distinguirlos?",
                    "Sí: el {item} es {color} y el {other_item} es {other_color}.",
                ),
                (
                    "¿De qué color es el {other_item}?",
                    "El {other_item} es {other_color}.",
                ),
                (
                    "Ahora quiero el color del {item}, sin comparaciones. ¿Cuál es?",
                    "El {item} es {color}.",
                ),
            ],
        },
        {
            "templateId": "current-question-two-people-and-activities",
            "turns": [
                (
                    "En una historia ficticia, {name} quiere {activity} y {friend} quiere {other_activity}. ¿Quién quiere hacer cada cosa?",
                    "{name} quiere {activity} y {friend} quiere {other_activity}.",
                ),
                (
                    "Pregunto sólo por {name}. ¿Qué quiere hacer?",
                    "{name} quiere {activity}.",
                ),
                (
                    "Deja a {name} aparte. ¿Qué quiere hacer {friend}?",
                    "{friend} quiere {other_activity}.",
                ),
            ],
        },
    ],
    "style_language": [
        {
            "templateId": "style-language-a-close-invitation",
            "turns": [
                (
                    "Voy a preparar un texto sobre {activity}. ¿Puedes ayudarme en español?",
                    "Sí. Podemos escribirlo en español.",
                ),
                (
                    "Quiero un tono cercano, sin palabras complicadas. ¿Cómo lo harías?",
                    "Usaría palabras sencillas y un tono natural.",
                ),
                (
                    "Sin comentarios, ¿qué única frase usarías para invitar a {friend} a {activity}?",
                    "{friend}, ¿te apetece {activity}?",
                ),
            ],
        },
        {
            "templateId": "style-language-formal-to-informal",
            "turns": [
                (
                    "Necesito pedir prestado el {item} a {friend}. ¿Podemos redactar esa petición?",
                    "Sí. Podemos pedir el {item} con una frase breve.",
                ),
                (
                    "Usa usted y un tono amable. ¿Qué pregunta escribirías?",
                    "¿Podría prestarme el {item}, por favor?",
                ),
                (
                    "Mejor tutea y evita una explicación. ¿Cómo queda la pregunta?",
                    "¿Me prestas el {item}, por favor?",
                ),
            ],
        },
        {
            "templateId": "style-language-an-answer-without-a-preface",
            "turns": [
                (
                    "Para una puesta en escena elegí el día {day} y el color {color}. ¿Puedes recordarlos en esta charla?",
                    "En esta charla, el día elegido es {day} y el color es {color}.",
                ),
                (
                    "Respóndeme siempre en español y ve al punto. ¿Qué color elegí?",
                    "Elegiste {color}.",
                ),
                (
                    "Sin introducir la respuesta, ¿qué día elegí? Dame sólo el día.",
                    "El {day}.",
                ),
            ],
        },
        {
            "templateId": "style-language-a-statement-to-an-invitation",
            "turns": [
                (
                    "Quiero mandar un mensaje sobre {activity} a {name}. ¿Puedes escribir con un tono natural?",
                    "Sí. Podemos usar una frase sencilla sobre {activity}.",
                ),
                (
                    "El mensaje dice que me apetece {activity}. ¿Cómo sonaría en español?",
                    "Me apetece {activity}.",
                ),
                (
                    "¿Cómo lo dirías como invitación para los dos, en una sola pregunta breve?",
                    "¿Te apetece {activity} conmigo?",
                ),
            ],
        },
    ],
}
