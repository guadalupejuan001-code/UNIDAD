# Base propia de 15M: respaldo intermedio

`step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86` conserva los cinco archivos
de un checkpoint nativo cerrado de la corrida causal propia iniciada desde
pesos aleatorios. Contiene nueve partes de como máximo 32 MiB; sus manifiestos
identifican hashes y tamaños para restaurar exactamente pesos, tokenizador,
estado y Adam/scheduler/RNG.

Es un respaldo de step 3.072, con 182.406.100 bytes originales. No representa
el cierre del entrenamiento, una aceptación conversacional ni una sustitución
de C7. La documentación y las instrucciones están en
[checkpoint-packaging.md](../../../docs/learning-repair-20261013/checkpoint-packaging.md).
