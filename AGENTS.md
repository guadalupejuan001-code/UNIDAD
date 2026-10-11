# Desarrollo de Unidad

Lee [ORDENES_USUARIO.md](ORDENES_USUARIO.md) antes de modificar el proyecto.
Ese documento conserva el objetivo y las instrucciones expresas del creador;
las nuevas instrucciones del usuario en la sesión tienen prioridad.

- Construye Unidad como un proyecto propio orientado a realizar trabajo útil por
  su creador. Los modelos pequeños actuales son componentes experimentales.
- Mantén el núcleo independiente de las interfaces: varios productos pueden
  conectarse a los mismos modelos con herramientas y memoria configuradas por
  producto. El contexto no debe permitir elegir rutas ni herramientas nuevas.
- Los flujos explícitos de operaciones no son planificación aprendida. Informa
  los pasos completados cuando falle una ejecución; no prometas revertir efectos
  de herramientas externas.
- Conserva el enfoque de entrenamiento desde pesos aleatorios. No sustituyas
  silenciosamente los modelos propios por llamadas a otra IA.
- Prioriza la conversación en español y el estilo cercano solicitado. Entrenar
  debe producir pesos y métricas verificables; no inventes respuestas para aparentar
  mejoras del modelo. Un entrenamiento largo debe ser reanudable y mostrar su
  estado real, incluido cuando el entorno deje de estar activo.
- Conserva conjuntos de evaluación separados por conversación y familia de
  ejemplos; no ajustes el modelo usando los resultados de test.
- Al ampliar el corpus, conserva las particiones anteriores y la procedencia de
  los pesos propios. Una etapa nueva puede reutilizar modelo y tokenizador; una
  reanudación mantiene además el optimizador. No confundas esas dos operaciones.
- Mantén objetivos y correcciones en texto libre. No inventes categorías de
  personalidad para representar al creador.
- Distingue registrar memoria, entrenar parámetros y ejecutar un trabajo. No
  atribuyas capacidades que no se hayan comprobado ni declares el objetivo
  completo cumplido por tener archivos de pesos o pruebas técnicas aprobadas.
- Mantén el código simple. Justifica las optimizaciones con mediciones y verifica
  que conservan los resultados que deban mantenerse.
- Usa el contrato conversacional común y `ConversationEngine` al conectar CLI,
  cerebro o evaluación; evita duplicar límites y caminos de generación. Conserva
  las reglas guardadas al reanudar checkpoints anteriores.
- Conserva los datos personales en las rutas privadas excluidas de Git. Adjuntar
  órdenes al repositorio no equivale a usarlas como corpus de entrenamiento.
- Para cambios de código, ejecuta las pruebas pertinentes. Para cambios sólo de
  documentación, comprueba el diff y los enlaces sin añadir pruebas redundantes.
