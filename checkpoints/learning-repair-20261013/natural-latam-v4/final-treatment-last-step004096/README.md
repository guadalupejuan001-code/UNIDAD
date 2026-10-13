# LAST final del tratamiento LATAM v4

El paquete [step-000004096-b39d6d921c7740c48a289e62d715654e](step-000004096-b39d6d921c7740c48a289e62d715654e/) conserva 182,360,596 bytes
nativos mediante nueve partes de hasta 32 MiB. Mantiene pesos, Adam/RNG, estado,
configuración, scheduler, orden y los nombres originales BEST/LAST. Todos los
hashes de partes y archivos completos fueron comprobados.

El BEST nativo completo permanece en la corrida privada original. El artefacto
BEST público reutiliza los pesos existentes del paso 1.024 y no contiene el
optimizador de ese checkpoint. La recuperación de la corrida completa requiere
resolver también ese puntero nativo BEST.

La [entrega final](../../../../docs/learning-repair-20261013/natural-latam-v4-treatment-final-delivery.md)
conserva instrucciones, hashes y límites. No se ejecutó restauración, inferencia,
pruebas ni entrenamiento durante este empaquetado.
