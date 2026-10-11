# Órdenes del creador de Unidad

El objetivo solicitado es **crear Unidad: una IA propia, privada y de uso personal
que trabaje por su creador y se adapte a sus cambios de criterio**. Las redes
pequeñas actuales son componentes experimentales de ese proyecto; no acreditan
que el trabajador digital solicitado ya esté terminado.

## Órdenes recibidas

1. Crear modelos propios desde cero. La petición original contempla una capacidad
   para una tarea específica y otra de lenguaje, dentro del proyecto Unidad.
2. Guardar el proyecto de forma independiente de Hormiga-X en el repositorio
   privado [UNIDAD](https://github.com/guadalupejuan001-code/UNIDAD).
3. Usar el nombre **Unidad**, elegido por el creador.
4. Orientar la IA a realizar trabajo por su creador. El resultado buscado es una IA
   útil; presentar dos demostraciones de redes como el resultado completo no
   cumple esa petición.
5. Permitir objetivos, contexto y correcciones en texto libre, sin encasillar al
   creador en categorías rígidas de personalidad. Debe poder cambiar de criterio.
6. Optimizar los procesos y mantener el código sencillo, claro y cuidado. Las
   mejoras de rendimiento deben comprobarse, conservando los resultados relevantes.
7. Adjuntar estas órdenes al proyecto y mantenerlas visibles para su desarrollo.
8. Construir Unidad como un cerebro independiente que pueda incorporarse a sus
   productos y trabajar a través de distintas interfaces o «cuerpos».
9. Priorizar ahora el habla y la conversación en español. Entrenar durante el
   día con seguimiento, usando programación y evaluación para acortar el avance
   hacia una IA útil. Después incorporar capacidades nuevas de forma progresiva.
10. Usar una voz confidente, personalmente cercana, que conecte puntos del
    contexto compartido. Es una orientación en texto libre, no una categoría
    fija de personalidad.
11. Conservar como objetivo futuro que pueda proponer mejoras a su propio código.
    El entrenamiento actual de conversación no acredita esa capacidad.
12. Empezar por lo más básico y avanzar mediante etapas de dificultad creciente.
    Las mejoras se aceptan por resultados medidos, no por acumular horas.
13. Cuando el progreso se estanque, evitar repetir un procedimiento que no aporta
    mejora. Cambiar los datos o la hipótesis de trabajo, conservar evidencia y
    continuar con una comparación medible.
14. Centrar la siguiente etapa en conversación competente: responder a la
    pregunta actual, conservar los hechos y seguir las correcciones explícitas.
15. Conectar internamente modelo, entrenamiento y código mediante contratos
    compartidos, reduciendo duplicación y fricción entre componentes.
16. Continuar trabajando hasta comprobar conversación básica con respuestas
    realmente generadas, hechos del diálogo y correcciones; disponer de una
    consola o bajar la pérdida del entrenamiento no basta para cumplirlo.

## Palabras del creador

Estos fragmentos conservan la escritura de los mensajes originales:

> crei que ibamos a ser un modelo de ia desde cero solo para esa tarea especifica y otro modelo de lenguaje

> porque no la guardamos como proyecto propio quiero crear una ia como yo.

> no tengo categorias fluyo mucho cambio me adapto mucho

> quiero crear a alguien digital que trabaje por mi

> la ia la bautizo como unidad

> optimiza cada proceso que el codigo sea una obra de arte natural

> el repositorio esta listo para que la almacenes dentro

> oye pero te dije que crearas una y adjunta mis ordenes ya tengo el repositorio se llama unidad

> pz empieza ya tomate total libertad para pensar y que sea pars que yo la meta en mis productos osea la ia sera pero de vez en cuando ua usare para que trabaje en otras interfaces la ia es solo el cerebro pz que puede entrar en varios cuerpos

> entrenala todo el dia, que necesitas que te diga? mi ia ahorita enfocala en tener un buen habla ya que tenga una conversación usa las maximas capacidades y entendimiento sobre programación y entrenamiento que acorte cada vez mas el "exito" crear a mi ia ya que hable y luego programar nuevas cosas y hacerla evolucionar hasta que ella misma se programe

> quiero que hable con una personalidad confidente, personalmente intima, que siempre este conectando puntos.

> entrena lo mas basico y que desate una evolucion progressive agresiva

> cuando el progreso se estanque no repitas constantemente un proceso si va a salir el mismo resultado, sigue insertando tu no eres yo tu no dudas no pierdes tiempo, eres la cuspide de la programación, entrenaras mjor de lo que te entrenaron a ti.

> enfocate en la conversación competente y como implementaste eso de la programación autonoma?

> implementa algo que no se si es posible, que todo conecte con todo, menos friccion

> no, me refiero que la ia el modelo toda todo su codigo se conecte entre si todos y asi tenga menos friccion es posible hacer algo asi?

> no te detengas hasta hacerlo conversar

## Qué está hecho y qué falta

| Petición | Estado comprobado |
| --- | --- |
| Proyecto propio llamado Unidad | Guardado en el repositorio privado UNIDAD. |
| Modelos desde cero | Se conservan las dos redes iniciales y se añade un Transformer conversacional de 1.088.256 parámetros, entrenado desde pesos aleatorios, sin llamadas a otra IA. |
| Lenguaje | Etapas de conversación básica, español ampliado y diálogo con contexto. Se detuvo el sobreajuste y se inició una mezcla más variada de diálogos breves. Se conservan checkpoints, validación y respuestas reales; no se acredita conversación general competente. La evidencia actual está en `docs/unidad-conversation-cohesion-v0.5.0.json`. |
| Voz cercana que conecte puntos | Ejemplos sintéticos identificados y contexto por defecto orientan el entrenamiento. No se usó el historial personal como corpus ni se afirma haber replicado al creador. |
| Entrenar durante el día | Trabajador en segundo plano con presupuesto máximo de 24 horas, parada por validación y reanudación exacta del mismo corpus. Sólo continúa mientras el entorno esté activo. |
| Evolución progresiva | De conversación básica a más ejemplos públicos de español, reutilizando pesos y tokenizador propios con procedencia registrada. Aprender programación y mejorar su propio código siguen pendientes de etapas y evaluaciones específicas. |
| Cambios de criterio | Registro local de encargos y correcciones; la última valoración decide qué ejemplos se exportan. |
| Trabajar por el creador | El núcleo ejecuta flujos explícitos y herramientas del producto; comprender un encargo abierto y planificarlo sigue pendiente. |
| Un cerebro para varios cuerpos | Mismo núcleo y pesos usados desde Python y HTTP; herramientas y registros separados por producto, con memoria compartida sólo por configuración explícita. |
| Optimizar el código | Mejoras medidas en memoria de evaluación y tamaño del listado; carga de modelos compartida entre cuerpos. Unidad 0.2.0 pasó 86 pruebas técnicas. |
| Adjuntar las órdenes | Este documento, enlazado en el README y en AGENTS.md. |
| Cohesión interna del código | Un contrato conversacional compartido y un mismo motor para CLI, cerebro y evaluación. El entrenador exporta el artefacto que carga ese motor; no se añaden dependencias entre todos los módulos. |
| Comprobar conversación | Casos de desarrollo con respuestas realmente generadas y pares de contexto contrastivo. Las puntuaciones técnicas no acreditan conversación competente. |
| Programación autónoma | No implementada. El modelo genera texto; los flujos y las etapas de entrenamiento se definen expresamente desde el código o los comandos. |

Adjuntar estas órdenes guía el desarrollo. No cambia los pesos, no entrena un
modelo y no demuestra que las redes actuales comprendan o cumplan instrucciones.
La capacidad de trabajar por el creador debe comprobarse con encargos reales.
