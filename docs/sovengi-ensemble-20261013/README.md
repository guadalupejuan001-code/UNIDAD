# Coordinación de modelos de SOVENGI, 13 de octubre de 2026

SOVENGI 0.8 conecta los 35 Transformers propios que existen en el repositorio.
La configuración anterior seleccionaba uno para generar y dejaba los demás
como referencias. Ahora cada solicitud genera varios candidatos mediante el
mismo `ConversationEngine`, compara sus textos y devuelve exactamente la salida
de un participante, con su `modelId` real. No se cambian ni se combinan pesos.

## Ejecución y límites

`configs/gnosis.json` conserva `reference-wiki1` como ancla principal y usa
`memory-candidate` como segunda ancla. Ambas admiten 2.048 tokens de contexto;
los otros dos puestos rotan por el catálogo disponible. Una vuelta cubre los
33 miembros restantes en 17 solicitudes. Con un límite de dos candidatos sólo
participan las anclas. La configuración alternativa invierte el ancla principal
y mantiene el catálogo común.

Los valores predeterminados son cuatro candidatos, dos éxitos mínimos, cuatro
motores residentes, 256 tokens nuevos en total, ocho puntuaciones y dos jueces.
El máximo solicitado por candidato se reparte si excede el presupuesto global.
Se reservan tokens antes de generar; una inferencia fallida conserva su cuota
completa cuando su coste no puede verificarse. La caché LRU libera referencias
a motores expulsados y reutiliza pesos residentes. El límite de 60 segundos
se comprueba entre operaciones: una inferencia en curso puede excederlo y la
traza declara el exceso. No es una interrupción estricta ni un SLA.

Cada participante prepara el contexto con su propio tokenizador y capacidad.
Conserva el sistema y el último mensaje, reserva la respuesta y elimina turnos
anteriores completos si es necesario. Si el mínimo no cabe, se registra el
fallo; no se corta silenciosamente el sistema. Los archivos ausentes o las
identidades/cabeceras inválidas quedan visibles y fuera de la rotación. La carga
real valida hashes y tensores con el cargador existente. Menos de dos candidatos
válidos produce un error con la traza del trabajo realizado.

## Selección y conexiones

Los jueces puntúan cada candidato con su propio tokenizador. Se ordena la NLL
por carácter dentro de cada juez completo y se agregan sus posiciones relativas;
no se comparan pérdidas brutas de vocabularios distintos. El conjunto de jueces
se decide antes de ordenar las cargas por residencia. Se añade soporte léxico
de las fuentes originales con peso 0,25. Estos criterios no verifican hechos ni
comprensión. La salida seleccionada no se reescribe ni se sustituye por notas.

La RNN propia puntúa un prefijo de hasta 512 caracteres como diagnóstico y
desempate, con su identidad y cobertura. No acredita verdad ni seguimiento de
instrucciones. El MLP conserva su dominio de visitas públicas de Wikimedia:
`conversation.forecast` recibe observaciones explícitas, calcula pronósticos,
adjunta evidencia acotada y pasa por contexto, memoria y coordinación. No
inventa observaciones ni usa el MLP para estimar recursos de ejecución.

Brain, GNOSIS, CLI y HTTP comparten el motor coordinado. La autonomía consulta
`prompt_usage` del motor en lugar de acceder a un Transformer privado. Los
comandos `learning-status` y `export` conectan las etapas GNOSIS ya existentes;
el historial privado no inicia aprendizaje automático. Las rutas y el catálogo
se configuran en el anfitrión, nunca mediante el texto de una solicitud.

## Evidencia

- [Catálogo y hashes verificados](model-catalog.json): 35 identidades distintas,
  pesos propios, arquitecturas y referencias históricas; los alias de checkpoints
  no se cuentan como modelos adicionales.
- [Reconstrucción del delta](delta-reconstruction.json): mismo SHA-256 previsto,
  pesos completos publicados para inferencia sin reconstrucción ni red.
- [Benchmark de caché](benchmark.json): artefactos pequeños propios, iguales
  respuestas y jueces con uno o varios motores residentes. No extrapola al
  rendimiento del catálogo de pesos completos.
- [Ejecución real y cobertura](real-evaluation.json): solicitudes ficticias,
  textos originales, participantes, costes y límites de evaluación.
- [Auditoría de servicios](../sovengi-connectivity-20261013/audit.json): conexiones
  anteriores y conexiones implementadas, con funciones y dominios reales.

La CI usa una descarga parcial para evitar todos los archivos históricos. El
catálogo sigue enumerando 35 y declara cuáles están instalados; los disponibles
participan sin presentar a los ausentes como ejecutados. Una copia completa del
repositorio permite toda la rotación. No se descarga ningún modelo durante una
respuesta. Ningún artefacto del catálogo ha superado la aceptación general de
conversación; la integración técnica y la calidad aprendida se evalúan aparte.
