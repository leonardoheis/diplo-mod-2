# Conclusiones Personales — Detección de Barcos en Imágenes Satelitales

**Alumno:** Maximiliano Miro  
**Cátedra:** Computer Vision — Diplomatura en Inteligencia Artificial  
**Fecha:** Junio 2026

> Este documento complementa el informe grupal (`conclusions.md`) con la experiencia personal de ejecutar las notebooks del proyecto: el detector clásico con OpenCV, los clasificadores CNN y MobileNetV2, y el fine-tuning de YOLO11.

---

## 1. Contexto de ejecución

Dentro del grupo, cada integrante trabajó con recursos tecnológicos distintos. En mi caso, **no cuento con placa aceleradora de video (GPU)** en mi PC local, por lo que para el entrenamiento de modelos tuve que recurrir a Google Colab.

Sin embargo, en lugar de usar Colab desde la interfaz web de Google, decidí aprovechar la **extensión de Colab para Visual Studio Code / Cursor**, que permite ejecutar notebooks usando el kernel de Colab directamente desde el IDE. Esto me permitió acceder a las herramientas de IA de Cursor para correcciones y mejoras de código durante la ejecución, lo que resultó muy útil.

La experiencia no estuvo exenta de dificultades:

- El manejo de **referencias de archivos y credenciales** es más parecido a trabajar en Colab puro que en un proyecto local: las rutas, los secrets y las APIs se comportan como si estuvieras en la interfaz de Google.
- La extensión es **menos estable** que usar Colab directamente en el navegador: hubo sesiones que se desconectaron inesperadamente durante el entrenamiento.
- A pesar de estos inconvenientes, creo que es una **buena opción para quienes no tienen GPU** y cuentan con una suscripción a un agente de código como Cursor o Claude.

En cuanto a las GPUs disponibles en Colab, pude probar distintas opciones. La **GPU T4** resultó ser una opción muy rápida para el fine-tuning de modelos, con tiempos de entrenamiento de ~18 minutos para 100 epochs.

---

## 2. Etiquetado del dataset

El proceso de etiquetado fue una de las partes más laboriosas y donde cometimos el **error más costoso del proyecto**.

Utilizamos **Roboflow** con el dataset de Kaggle (*Ships in Satellite Imagery* de rhammell), al que luego agregamos nuevas imágenes de un dataset complementario. El etiquetado se hizo manualmente usando las herramientas de Roboflow.

El error fue el siguiente: **comenzamos etiquetando con segmentación en lugar de bounding boxes**. YOLO11 no puede entrenar con anotaciones de segmentación en modo detección, por lo que tuvimos que revisar y corregir las anotaciones. Ultralytics emitía el warning *"mixed detect-segment dataset"* durante el entrenamiento, indicando que quedaban algunas máscaras sin convertir, lo que afectó levemente la calidad de supervisión en las primeras corridas.

**Aprendizaje clave:** antes de anotar, es fundamental definir el formato de salida del modelo (detección, segmentación o clasificación) y usar el tipo de anotación correcto desde el inicio. Cambiar de segmentación a boxes una vez etiquetadas cientos de imágenes genera trabajo extra y ruido en los datos.

---

## 3. Detector clásico con OpenCV

### 3.1 Primera versión

La primera implementación del detector clásico (`satellite_ship_pipeline.ipynb`) usó una estrategia multicanal:

1. **Canal V del espacio HSV**: detecta objetos más brillantes que el agua.
2. **Canal de saturación**: detecta barcos de colores contrastantes (rojo, naranja, blanco).
3. **Operaciones morfológicas**: cierre para unir regiones fragmentadas, apertura para eliminar ruido.
4. **Filtro de área y aspect ratio**: los barcos son objetos elongados (aspect ratio > 1.5).

El resultado fue parcial: el modelo detectaba barcos en algunas imágenes, pero fallaba en muchas otras.

### 3.2 Mejoras implementadas

Para mejorar los resultados, incorporamos las siguientes técnicas adicionales:

- **CLAHE + Otsu**: contraste adaptativo local antes del umbralado, para mejorar la detección en imágenes con iluminación no uniforme.
- **Detección por bordes con Canny**: para capturar siluetas de barcos que no se distinguen bien por color.
- **Morfología asimétrica**: apertura 3×3 / cierre 5×5 / dilatación, para mejor manejo de formas irregulares.
- **Exclusión por color en espacio LAB**: descartar píxeles que corresponden al agua según su color, reduciendo falsos positivos.
- **NMS (Non-Maximum Suppression)**: para eliminar detecciones duplicadas solapadas.

Con estas mejoras logramos detectar la mayoría de los barcos en las imágenes de prueba. Sin embargo, el comportamiento seguía siendo **inconsistente entre imágenes**: los parámetros calibrados funcionaban bien en algunas escenas y fallaban en otras con condiciones lumínicas o de fondo distintas.

### 3.3 Por qué el enfoque clásico tiene un techo bajo

El detector clásico no aprende qué es un barco; solo captura propiedades espectrales y geométricas locales. Eso funciona bien cuando las condiciones son similares a las de calibración, pero no generaliza. La variabilidad fotométrica de las imágenes satelitales (distintas horas, cobertura nubosa, profundidad del agua, tipo de puerto) hace que cualquier conjunto fijo de parámetros tenga casos donde falla.

---

## 4. Clasificadores neuronales: CNN y MobileNetV2

Para construir un clasificador de chips, trabajamos con el dataset *Ships in Satellite Imagery* de Kaggle (~4000 tiles de 80×80 px etiquetados como *ship* / *no-ship*).

### 4.1 Preprocesamiento

- Split **70/15/15** (train / val / test) con `stratify` para mantener proporciones de clases.
- Normalización de píxeles a [0, 1].
- Data augmentation con `torchvision.transforms`:
  - Flip horizontal y vertical.
  - Rotación aleatoria ±20°.
  - Affine: traslación ±10%, escala ±10%.

### 4.2 CNN Baseline (arquitectura propia)

Red convolucional entrenada desde cero sobre los chips de 80×80 px. Entrenamiento por **25 epochs con early stopping** (patience=5), usando `CrossEntropyLoss` y optimizador `Adam`.

### 4.3 MobileNetV2 con Transfer Learning

Se usó **MobileNetV2 preentrenado en ImageNet** con entrenamiento en dos fases:

- **Fase 1**: Solo se entrena la cabeza de clasificación (base congelada).
- **Fase 2**: Se descongelan los últimos 5 bloques de features y se fine-tunea con learning rate reducido.

### 4.4 Resultados y conclusión comparativa


| Criterio         | CNN Baseline | MobileNetV2 |
| ---------------- | ------------ | ----------- |
| Recall (ship)    | **0.980**    | 0.893       |
| Precisión (ship) | 0.886        | **0.964**   |
| Accuracy         | ~96%         | ~96%        |
| AUC-ROC          | **0.9945**   | 0.9933      |


La CNN baseline supera a MobileNetV2 en recall (0.98 vs 0.89), lo que la hace más adecuada para vigilancia marítima donde no detectar un barco es peor que una falsa alarma. MobileNetV2 domina en precisión.

Ambos modelos alcanzan accuracy ~96%, lo que confirma que los chips de 80×80 px contienen suficiente señal para clasificación. Sin embargo, **no proveen coordenadas de bounding box**, lo que los limita frente a YOLO11 para detección real en imágenes completas.

---

## 5. Fine-tuning YOLO11

### 5.1 Configuración del entrenamiento


| Parámetro   | Valor        | Justificación                                           |
| ----------- | ------------ | ------------------------------------------------------- |
| `model`     | `yolo11m.pt` | Balance entre capacidad y velocidad para GPU de Colab   |
| `epochs`    | 100          | 30 epochs resultó insuficiente (underfitting observado) |
| `imgsz`     | 640          | Compatible con el tamaño de los chips de entrenamiento  |
| `cls`       | 1.5          | Mayor penalización en errores de clasificación          |
| `degrees`   | 30.0         | Barcos en cualquier orientación                         |
| `mosaic`    | 1.0          | Augmentación por mosaico activa todo el entrenamiento   |
| `optimizer` | auto (AdamW) | Determinado automáticamente por Ultralytics             |
| `cos_lr`    | True         | Cosine decay para mejor convergencia                    |


### 5.2 Múltiples iteraciones del entrenamiento

El entrenamiento requirió **múltiples ejecuciones** hasta llegar al run definitivo (`ship_detection_v1-9`). Las desconexiones de sesión en Colab (especialmente al usar la extensión de VS Code) interrumpieron varias corridas antes de completar los 100 epochs.

La **persistencia en Google Drive** resultó fundamental aquí: al guardar el dataset descargado, los pesos base y todos los runs en Drive desde el inicio, no fue necesario re-descargar nada entre sesiones. Retomar el trabajo luego de una desconexión era cuestión de segundos.

### 5.3 Resultados

**Validación al finalizar el entrenamiento (epoch 100):**


| Métrica         | Valor |
| --------------- | ----- |
| Precisión (val) | 0.817 |
| Recall (val)    | 0.496 |
| mAP@50 (val)    | 0.546 |
| mAP@50-95 (val) | 0.292 |


**Evaluación sobre el test set:**


| Métrica            | Valor          |
| ------------------ | -------------- |
| mAP@50             | **0.474**      |
| mAP@50-95          | 0.228          |
| Precisión          | 0.753          |
| Recall             | **0.459**      |
| Velocidad (GPU T4) | 25.9 ms/imagen |


### 5.4 Análisis del umbral de confianza

Realicé un barrido sobre 20 imágenes de test con distintos valores de `conf`. El resultado:

- **conf < 0.20**: más detecciones, pero alta tasa de falsos positivos.
- **conf 0.25–0.40**: balance razonable para vigilancia marítima.
- **conf > 0.50**: caída abrupta; se pierden barcos reales.

Para vigilancia marítima, donde **no detectar un barco es peor que una falsa alarma**, se recomienda operar con **conf = 0.25**.

---

## 6. Reflexiones finales

### Lo que más impactó en los resultados

1. **El error de etiquetado con segmentos** fue un error que me retrasó un poco. Corregir anotaciones después del hecho es tedioso y deja ruido residual en el dataset.
2. **El recall bajo de YOLO11** (0.46) no es un fallo del modelo, sino del dataset: los chips de entrenamiento son homogéneos (barcos bien centrados, fondo limpio). Los casos difíciles (barcos pequeños, zonas portuarias densas) están sub-representados. A esto se suma que el modelo confunde estructuras costeras, plataformas, docks, olas y reflexiones con barcos — exactamente los falsos positivos que aparecen en las zonas más complejas. Ambos problemas tienen la misma raíz: falta de diversidad en los datos de entrenamiento.
3. **La extensión de Colab para VS Code/Cursor** es poderosa pero frágil. Las desconexiones inesperadas durante el entrenamiento son el mayor costo de usarla en lugar de la UI de Google.

### Qué haría diferente

- **Definir el tipo de anotación antes de empezar a etiquetar.** El error de usar segmentos en lugar de boxes se puede evitar completamente con una decisión de diseño previa.
- **Usar Drive como caché desde el primer notebook**, no como una mejora posterior. Las desconexiones de Colab son inevitables y la persistencia elimina la mayor parte de la fricción.
- **Entrenar con más epochs (150–200)**: las curvas muestran que el mAP en validación seguía creciendo lentamente al llegar a epoch 100 y no se activó el early stopping.
- **Aplicar Hard Negative Mining**: el modelo confunde estructuras costeras, plataformas, docks, olas y reflexiones con barcos. La solución es agregar al dataset tiles sin barcos que contengan esas estructuras engañosas — cuando el modelo las ve durante el entrenamiento, aprende a clasificarlas correctamente como fondo. En Roboflow habría que cargar imágenes del tipo "fondo difícil" sin anotaciones, como clase *background*. Esta mejora atacaría directamente los falsos positivos más frecuentes sin necesidad de cambiar la arquitectura ni los hiperparámetros.
- **Implementar SAHI para inferencia sobre escenas completas**: el notebook lo lista como próximo paso pero no lo implementé. La experiencia del grupo (documentada en `conclusions_LJ.md`) confirma que el slicing transforma "cero detecciones" en detecciones reales sobre imágenes satelitales de alta resolución con el mismo modelo entrenado.

---

## Referencias

- Ultralytics YOLO11: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- Dataset Kaggle shipsnet: [https://www.kaggle.com/rhammell/ships-in-satellite-imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery)
- SAHI para objetos pequeños: [https://github.com/obss/sahi](https://github.com/obss/sahi)
- Conclusiones grupales del proyecto: `conclusions.md`
- Conclusiones personales de Leandro Juárez: `conclusions_LJ.md`

