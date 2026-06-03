# Guía de conceptos — Exposición de Computer Vision
**Detección de Barcos en Imágenes Satelitales · Leonardo Heis, Leandro Juárez, Maximiliano Miro**

> Cada sección explica el concepto teórico y lo ancla en nuestros resultados concretos. Usarlo como guía de repaso, no para leer en voz alta.

---

## 1. Bounding box — ¿qué es y cómo se representa?

### Qué es

Un **bounding box** (caja delimitadora) es el rectángulo mínimo que encierra un objeto detectado en una imagen. Es la unidad de salida de cualquier detector de objetos: en lugar de decir "hay un barco en esta imagen", el modelo dice "hay un barco **acá**, en estas coordenadas exactas".

### Las dos representaciones más comunes

```
Formato absoluto (píxeles)       Formato YOLO (normalizado 0–1)
┌──────────────────────────┐     ┌──────────────────────────┐
│                          │     │                          │
│      (x1, y1)            │     │     clase  cx   cy  w  h │
│        ┌────────┐        │     │        0  0.5  0.5 0.45 0.45
│        │  barco │        │     │                          │
│        └────────┘        │     │  cx, cy = centro / ancho │
│              (x2, y2)    │     │  w, h   = tamaño / ancho │
└──────────────────────────┘     └──────────────────────────┘
  esquina sup-izq + inf-der         centro + dimensiones
```

**YOLO usa el formato normalizado.** Todos los valores están entre 0 y 1, lo que hace que las coordenadas sean independientes de la resolución de la imagen.

### En nuestro proyecto

Cuando convertimos los chips de ShipsNet (clasificación) a formato YOLO (detección), usamos la aproximación `0 0.5 0.5 0.45 0.45`: el barco siempre está centrado en el chip, así que el centro es el punto medio y el tamaño es el 45% de cada dimensión. Es una aproximación burda pero funcional para bootstrapping.

Las anotaciones del dataset de Roboflow, en cambio, tienen bounding boxes ajustadas manualmente alrededor del casco real de cada barco — por eso ese dataset produce mejor recall.

---

## 2. mAP y la curva Precision-Recall

### Precisión y Recall: la tensión fundamental

Antes de entender mAP, hay que entender el trade-off que resume:

```
                    Lo que el modelo DICE que es barco
                    ┌─────────────────────────────────┐
                    │  Correcto (TP)  │  Error (FP)   │
 Lo que            ─┼─────────────────┼───────────────┤
 realmente          │  Perdido (FN)   │  (TN)         │
 es barco          ─┴─────────────────┴───────────────┘

  Precisión = TP / (TP + FP)  → "de lo que detecté, ¿cuánto era real?"
  Recall    = TP / (TP + FN)  → "de lo real, ¿cuánto detecté?"
```

Subir el **umbral de confianza** → sube Precisión, baja Recall (el modelo es más exigente, detecta menos cosas pero las que detecta son más seguras).

Bajar el **umbral de confianza** → sube Recall, baja Precisión (detecta más barcos, pero incluye más falsos positivos).

### La curva Precision-Recall

Trazando Precisión vs. Recall para todos los umbrales de confianza posibles se obtiene una curva. **El área bajo esa curva es el AP (Average Precision)** para una clase.

```
Precisión
  1.0 ┤───────╮
      │       ╰──╮
  0.7 ┤           ╰────╮
      │                 ╰───╮
  0.4 ┤                      ╰──────
      └─────────────────────────────
     0.0        0.5           1.0   Recall

  AP = área sombreada bajo la curva
  Modelo ideal → AP = 1.0 (curva pegada a la esquina superior derecha)
```

### mAP = promedio de AP sobre clases

`mAP@50` significa: AP calculado con umbral IoU = 0.50 (ver sección 3), promediado sobre todas las clases del dataset. Como nuestro dataset tiene una sola clase (`ship`), `mAP@50 = AP@50`.

`mAP@50-95` es el promedio de AP calculado con umbrales IoU de 0.50, 0.55, ..., 0.95. Es una métrica más exigente porque requiere que las cajas sean precisas.

### En nuestro proyecto

| Modelo | mAP@50 | mAP@50-95 | Precisión | Recall |
|--------|--------|-----------|-----------|--------|
| v1 (30 epochs) | 0.463 | — | 0.507 | 0.401 |
| v2 (100 epochs) | 0.616 | — | 0.748 | 0.468 |
| v3 (120 epochs, imgsz=960) | **0.611** | 0.333 | **0.748** | **0.469** |

La brecha entre mAP@50 (0.611) y mAP@50-95 (0.333) nos dice que el modelo **localiza bien** los barcos (los detecta dentro del área correcta con IoU > 0.5) pero **no ajusta perfectamente** los bordes de la caja (falla al exigir IoU > 0.75 o 0.90). Tiene sentido: los barcos son objetos pequeños y las anotaciones automáticas de Roboflow no son perfectas.

### Por qué el recall (0.47) es el número que más nos preocupa

Para vigilancia marítima, no detectar un barco real (falso negativo) es peor que una falsa alarma. Un barco de pesca ilegal que pasa desapercibido es un problema operativo; una falsa alarma es una verificación extra. Por eso operamos con `conf=0.20` en lugar del default de 0.25 — sacrificamos precisión para recuperar recall.

---

## 3. IoU — Intersection over Union

### Definición

IoU mide cuánto se superponen dos rectángulos: la predicción del modelo y la anotación real (*ground truth*).

```
  Ground truth        Predicción           Intersección ∩   Unión ∪
  ┌──────────┐        ┌──────────┐
  │          │        │    ┌─────┼───┐         ┌─────┐      ┌────────────┐
  │          │  +     │    │ ∩∩∩ │   │    →    │ ∩∩∩ │  /   │            │
  │          │        └────┼─────┘   │         └─────┘      │            │
  └──────────┘             └─────────┘                       └────────────┘

              IoU = Área(∩) / Área(∪)         rango: 0 (sin overlap) → 1 (caja perfecta)
```

### Uso 1: Evaluación de métricas

Al calcular si una detección es TP o FP, se compara la caja predicha con la ground truth. Si `IoU ≥ umbral`, es un True Positive. El umbral más común es **0.50** (mAP@50), que es generoso: acepta cajas que cubren al menos la mitad del objeto real.

```
IoU = 0.82   →  TP  (caja bastante ajustada)
IoU = 0.51   →  TP  (caja grande pero cubre bien el objeto)
IoU = 0.38   →  FP  (caja demasiado desplazada o de tamaño incorrecto)
```

### Uso 2: NMS — Non-Maximum Suppression

YOLO genera cientos de cajas candidatas por imagen. Sin filtrado, un mismo barco aparecería detectado 10 veces. **NMS** elimina los duplicados:

```
Paso 1: ordenar todas las cajas por confianza (de mayor a menor)
Paso 2: tomar la caja más confiable → mantenerla
Paso 3: calcular IoU entre esa caja y todas las demás
Paso 4: eliminar todas las cajas con IoU > umbral_NMS (son duplicados del mismo objeto)
Paso 5: repetir con la siguiente caja que quedó
```

El parámetro `iou=0.5` que usamos en `yolo predict` es este umbral de NMS. Un umbral bajo (0.3) es agresivo: elimina cajas que solo se tocan un poco, útil cuando los objetos están separados. Un umbral alto (0.7) es conservador: solo elimina cajas casi idénticas, útil cuando los objetos están muy juntos (como barcos en un puerto).

### En nuestro proyecto

Usamos `iou=0.45` para inferencia en el test set. Con barcos en zonas portuarias densas, un umbral muy bajo haría que el NMS elimine detecciones de barcos distintos que están lado a lado.

---

## 4. Train / Valid / Test — Por qué dividir y qué pasa si no

### La pregunta de fondo: ¿cómo sabemos si el modelo aprendió o memorizó?

Un modelo tiene millones de parámetros. Si lo entrenamos y evaluamos sobre los **mismos datos**, siempre va a parecer que funciona bien — simplemente "recuerda" las respuestas. Lo que nos interesa saber es si generaliza a datos que nunca vio.

### Los tres roles

```
  TRAIN (70%)          VALID (20%)              TEST (10%)
  ──────────────       ──────────────────       ──────────────────
  El modelo aprende    El entrenador aprende     El árbitro final
  de estos datos.      de estas métricas:        que nadie miró
                       ¿seguir entrenando?       durante el desarrollo.
                       ¿cambiar lr?
                       ¿cuándo parar?
```

**Valid** influye en las decisiones de entrenamiento (early stopping, selección del mejor checkpoint, tuning de hiperparámetros). Por eso, al final, también tiene información "filtrada" del modelo — el modelo indirectamente se adaptó a él. **Test** es el único set que no tocó ninguna decisión: mide el rendimiento real.

### ¿Qué pasa si no se divide?

| Escenario | Problema |
|-----------|----------|
| Evaluar en train | Métricas perfectas que no reflejan la realidad. El modelo memorizó, no aprendió. |
| Usar test para ajustar hiperparámetros | El test deja de ser imparcial. Estás "haciendo trampa" sin saberlo. |
| Sin valid set | No hay señal para early stopping. El modelo sobreajusta sin que nadie lo note. |
| Dataset pequeño sin división | Estimas mal ambas cosas: ni el ajuste ni la generalización son confiables. |

### En nuestro proyecto

| Split | Imágenes | Bounding boxes |
|-------|----------|----------------|
| Train | 1535 | 7488 |
| Valid | 512 | 2606 |
| Test | 512 | 2768 |

División 70/20/10 configurada en Roboflow. El early stopping (`patience=50`) monitoreó el mAP@50 en validación — cuando esa métrica dejó de mejorar por 50 epochs consecutivas, el entrenamiento se detuvo. El **test set nunca fue visto** hasta el reporte final de métricas.

---

## 5. Sobreajuste — qué es y cómo verlo en las curvas

### Qué es

Un modelo **sobreajusta** (*overfitting*) cuando aprende los datos de entrenamiento de memoria, incluyendo el ruido, en lugar de aprender los patrones generales. Funciona muy bien en train pero mal en datos nuevos.

La analogía: un estudiante que memoriza los ejercicios del práctico pero no entiende el concepto → en el examen con ejercicios nuevos, falla.

### El espectro completo

```
     SUBAJUSTE              AJUSTE CORRECTO              SOBREAJUSTE
   (underfitting)             (good fit)                 (overfitting)
   ───────────────         ─────────────────           ─────────────────
   El modelo es            Train y valid               El modelo "memoriza"
   demasiado simple,       mejoran juntos y            el ruido del train.
   no captura              convergen a valores
   el patrón.              similares.

   Señal: ambas curvas     Señal: ambas curvas         Señal: train sigue
   (train y val) están     bajan juntas y se           bajando, val empieza
   altas y no bajan.       estabilizan cerca.          a SUBIR (o diverge).
```

### Cómo verlo en las curvas de entrenamiento

```
Loss durante el entrenamiento

  SUBAJUSTE (v1, 30 epochs):       SOBREAJUSTE (hipotético):
                                   
  Loss │ train ──────────╲         Loss │ train ──────────────╲
       │ val   ──────────╲              │ val   ────╲    ╱────
       │                  ╲             │             ╲ ╱
       └──────────── epochs             └──────────── epochs
       Ambas siguen bajando             Val baja, luego SUBE
       al terminar → necesita           → el modelo sobreajustó
       más epochs

  AJUSTE CORRECTO (v2/v3):
  
  Loss │ train ─────────────╲
       │ val   ─────────────╲╲
       │                      ╲╲───── (convergen cerca)
       └──────────────────── epochs
       Val deja de bajar pero no sube → parar aquí
```

### En nuestro proyecto

Las curvas de v2 y v3 muestran **ajuste correcto** pero con señales de que el modelo llegó a su límite de datos:
- **Box loss en train:** 1.69 → baja continuamente.
- **Box loss en validación:** 1.93 → se estabiliza, brecha pequeña con train.
- **mAP@50 en validación:** crece hasta ~60 epochs y luego se estabiliza sin degradar.
- El early stopping (`patience=50`) no se activó → el modelo no sobreajustó, simplemente convergió lentamente.

La conclusión es que **no hay sobreajuste, hay subajuste de datos**: el modelo aprendió todo lo que pudo de las 1535 imágenes de train y llegó a un techo. Para romper ese techo, la solución es más datos, no regularización.

---

## 6. Detector clásico vs. YOLO — cuándo usar cada uno

### Resumen de qué hace cada enfoque

```
  DETECTOR CLÁSICO (OpenCV)          YOLO11
  ──────────────────────────         ────────────────────────────
  Imagen → HSV → umbral → morfología  Imagen → red neuronal → cajas
  → contornos → filtro de aspecto     + clase + confianza

  No aprende. Reglas escritas a mano.  Aprende de ejemplos anotados.
  Sin datos. Sin GPU. Sin entrenamiento.  Necesita datos, GPU, epochs.
```

### Cuándo preferir el detector clásico

| Situación | Por qué el clásico gana |
|-----------|-------------------------|
| No tenés datos etiquetados | YOLO necesita cientos de bounding boxes anotadas. El clásico no necesita nada. |
| Presupuesto computacional mínimo | Corre en CPU en tiempo real. YOLO necesita GPU para entrenar. |
| Prototipo en pocas horas | Implementación rápida para validar si el problema es abordable. |
| Condiciones muy controladas | Fondo uniforme, objetos de color contrastante, iluminación constante. El clásico es suficiente. |
| Interpretabilidad requerida | Cada decisión del pipeline es explícita: "filtré por saturación > X y área > Y". |

### Cuándo preferir YOLO

| Situación | Por qué YOLO gana |
|-----------|------------------|
| Alta variabilidad visual | Distintas condiciones de luz, fondos heterogéneos, objetos de cualquier color. |
| Necesitás coordenadas precisas | El clásico da contornos aproximados; YOLO da bounding boxes con coordenadas exactas. |
| Múltiples clases | Detectar barcos, boyas Y plataformas en la misma imagen en un solo paso. |
| Escenas complejas | Puertos con grúas, diques y barcos: el clásico no puede distinguirlos; YOLO sí con suficientes datos. |
| Mejor generalización | Calibrás los parámetros del clásico para una escena y falla en otra; YOLO aprende patrones que transfieren. |

### En nuestro proyecto: el clásico falló donde tenía que fallar

```
Detector clásico:
  ✓ Barcos rojos/naranjas sobre agua oscura → contraste alto, funciona
  ✗ Barcos grises sobre mar claro → contraste insuficiente
  ✗ Barcos en puerto con grúas → la morfología no distingue casco de estructura
  ✗ Barcos < 10px → quedan por debajo del umbral de área mínima

YOLO11m:
  ✓ Detecta barcos en zonas portuarias densas
  ✓ Independiente del color del casco
  ✓ Produce coordenadas precisas (mAP@50 = 0.61)
  ✗ Recall bajo en barcos muy pequeños (0.47) — problema de datos, no de arquitectura
```

### La lección del proyecto

El detector clásico fue valioso como **baseline rápido**: en dos horas validamos que el problema era real y que los barcos tenían propiedades espectrales distinguibles en algunos casos. Pero para una solución robusta en producción, donde las escenas son diversas y la localización importa, YOLO es el único enfoque end-to-end viable. La limitación no fue el modelo, sino los datos: con más imágenes anotadas y ejemplos negativos difíciles, el recall habría subido significativamente.

---

## Tabla resumen para repaso rápido

| Concepto | Una línea |
|----------|-----------|
| Bounding box | Rectángulo que localiza un objeto; en YOLO se representa como `(clase, cx, cy, w, h)` normalizados. |
| Precisión | De todo lo que detecté, ¿cuánto era real? `TP / (TP + FP)` |
| Recall | De todo lo real, ¿cuánto detecté? `TP / (TP + FN)` |
| AP | Área bajo la curva Precision-Recall para una clase y un umbral de IoU. |
| mAP@50 | Promedio de AP sobre todas las clases, con IoU ≥ 0.50 para contar un TP. |
| IoU | Solapamiento entre caja predicha y ground truth. `Intersección / Unión`. |
| NMS | Elimina cajas duplicadas sobre el mismo objeto usando IoU como criterio. |
| Train/Valid/Test | Train: aprender. Valid: ajustar sin contaminar. Test: medir sin sesgo. |
| Overfitting | El modelo memoriza train; val loss sube mientras train loss sigue bajando. |
| Underfitting | El modelo es demasiado simple o entrenó poco; ambas curvas se estancan altas. |
| Clásico vs YOLO | Clásico: rápido, sin datos, frágil. YOLO: robusto, necesita datos anotados y GPU. |
