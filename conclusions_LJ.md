# Conclusiones Personales — Detección de Barcos en Imágenes Satelitales

**Alumno:** Leandro Juárez  
**Cátedra:** Computer Vision — Diplomatura en Inteligencia Artificial  
**Fecha:** Junio 2026

> Este documento complementa el informe grupal (`conclusions.md`) con la experiencia personal de ejecutar las notebooks del proyecto, incluyendo las iteraciones propias sobre la detección en escenas satelitales de la Bahía de San Francisco.

---

## 1. Contexto de ejecución

Todas las ejecuciones descritas en este documento se realizaron en **Google Colab** con GPU T4, dado que no cuento con GPU dedicada en mi laptop. El entorno local es macOS con Apple Silicon, lo que limita el entrenamiento a MPS (Metal Performance Shaders) o CPU — considerablemente más lento y con menor capacidad de batch que una T4.

Esta restricción marcó todas las decisiones: tiempos de entrenamiento más largos en Colab, necesidad de mantener la sesión activa, y la motivación para crear una notebook adaptada al entorno macOS que permitiera al menos ejecutar inferencia y pasos de exploración sin depender de la nube.

---

## 2. Evolución de las notebooks: de cero detecciones a detección parcial con slicing

### 2.1 Primera versión — Sin detecciones en SF Bay

La primera notebook (`ship_detection_satellite_yolo11 - primera version (No detecta imágenes SF Bay).ipynb`) implementó el pipeline completo de entrenamiento con YOLO11m sobre los chips de ShipsNet, pero al ejecutar inferencia directa sobre las escenas satelitales completas de la Bahía de San Francisco, el resultado fue **cero detecciones**.

La causa raíz es un problema de escala: el modelo fue entrenado con chips de 80×80px donde el barco ocupa gran parte del encuadre, mientras que en una escena completa de ~2500×1700px, un barco representa apenas 10–15px de un total de más de cuatro millones de píxeles. La red nunca vio objetos tan pequeños en proporción durante el entrenamiento, por lo que directamente los ignora.

**Aprendizaje clave:** entrenar con chips pequeños y luego intentar inferencia sobre imágenes completas sin ninguna estrategia de escala es un error de diseño. El modelo aprende a reconocer patrones en una escala específica y no generaliza a escalas radicalmente distintas.

### 2.2 Segunda versión — Slicing con SAHI

La segunda notebook (`ship_detection_satellite_yolo11 - 2.ipynb`) incorporó **SAHI** (*Slicing Aided Hyper Inference*) para resolver el problema de escala. La estrategia consiste en:

1. Dividir la escena completa en tiles solapados del mismo tamaño que los chips de entrenamiento (320×320px con ~20% de overlap).
2. Ejecutar el modelo sobre cada tile individualmente.
3. Fusionar todas las detecciones con NMS para eliminar duplicados en los bordes de los tiles.

Este cambio produjo una mejora concreta: **el modelo comenzó a detectar barcos** en las escenas de SF Bay donde antes no detectaba nada. Sin embargo, el resultado no fue limpio: junto con las detecciones correctas aparecieron **falsos positivos** — regiones de agua con espuma, estructuras portuarias o reflejos que el modelo clasificó como barcos.

**Comparación entre versiones:**

| Aspecto | Primera versión | Segunda versión (SAHI) |
|---|---|---|
| Detecciones en SF Bay | 0 barcos | Varios barcos detectados |
| Falsos positivos | — | Presentes (espuma, diques) |
| Estrategia de escala | Inferencia directa | Slicing + NMS |
| Umbral de confianza | 0.25 | 0.30 |

### 2.3 Por qué aún hay falsos positivos después del slicing

El slicing resolvió el problema de escala, pero no el de distribución de datos. Los chips de entrenamiento provienen de ShipsNet, donde los barcos están siempre centrados en el chip y el fondo es homogéneo. Las escenas de SF Bay contienen:

- Estructuras portuarias (diques, grúas, muelles) con geometría rectangular similar a la de los cascos.
- Espuma y patrones de olas que generan texturas parecidas a los reflejos metálicos de los barcos.
- Variaciones de iluminación distintas a las del dataset de entrenamiento.

El modelo no fue entrenado con suficientes ejemplos negativos de estos casos particulares, por lo que no aprendió a distinguirlos con confianza. Para reducir los falsos positivos habría que incorporar chips de fondo específicos de SF Bay como ejemplos negativos durante el entrenamiento, o aumentar el umbral de confianza aceptando perder algunas detecciones reales.

---

## 3. Notebook adaptada para macOS

Dado que el entorno de trabajo principal es macOS con Apple Silicon, desarrollé la notebook `ship_detection_macos.ipynb` para poder ejecutar los pasos que no requieren GPU intensiva (exploración de datos, visualización, inferencia sobre pocas imágenes) sin depender de Colab.

La notebook detecta automáticamente el acelerador disponible:
- **MPS** (Apple Silicon) — más rápido que CPU, pero sin soporte para algunas operaciones de CUDA.
- **CPU** — fallback para operaciones no soportadas por MPS.

Las principales adaptaciones respecto a la versión de Colab:

| Aspecto | Colab (GPU T4) | macOS (MPS/CPU) |
|---|---|---|
| Credenciales | Colab Secrets | Archivo `.env` local |
| Batch size | 32 (CUDA) | 8 (MPS) / 4 (CPU) |
| Acelerador | `cuda:0` | `mps` o `cpu` |
| Tiempo de entrenamiento | ~15 min / 50 epochs | Inviable para >10 epochs |

El entrenamiento completo en macOS es imprácticamente lento (sin GPU CUDA), por lo que la notebook está orientada a inferencia y exploración. Los pesos entrenados en Colab pueden cargarse localmente para inferencia sin problema.

---

## 4. Reflexiones sobre el proceso

### Lo que más impactó en los resultados

El hallazgo más importante es que el **problema de escala** (no el modelo ni el dataset en sí) fue la causa principal de los resultados pobres en la primera versión. Con exactamente el mismo modelo entrenado, pasar de inferencia directa a inferencia con slicing transformó "cero detecciones" en "detecciones parciales con ruido". Esto confirma lo señalado en el informe grupal sobre SAHI como mejora de alto impacto.

### La limitación de no tener GPU local

Trabajar exclusivamente en Colab introduce fricciones reales: el entorno se reinicia si la sesión cae, los datasets hay que descargarlos cada vez (a menos que uses Google Drive como caché), y el tiempo de GPU disponible en la capa gratuita es limitado. Esto ralentizó las iteraciones y redujo la cantidad de experimentos que pude hacer. Una GPU local — aunque sea de gama media — habría permitido ciclos de experimentación mucho más rápidos.

### Qué haría diferente

- **Usar Google Drive como caché permanente** de los datasets descargados de Kaggle y los pesos entrenados, para no volver a descargarlos en cada sesión de Colab.
- **Entrenar con ejemplos negativos específicos de SF Bay.** Los falsos positivos del slicing son consistentes: siempre ocurren en zonas de espuma o estructuras portuarias. Agregar chips negativos de esas zonas al dataset de entrenamiento atacaría el problema en la raíz.
- **Explorar `conf=0.40–0.50`** para reducir falsos positivos en las escenas de SF Bay, asumiendo que se pierde algo de recall en barcos de baja confianza.
- **Integrar SAHI desde el comienzo** en la notebook principal, en lugar de como una sección tardía. Es fundamental para cualquier escena de resolución mayor a 640px.

---

## 5. Relación con las conclusiones grupales

El informe grupal identifica correctamente que *"aplicar SAHI es la mejora de mayor impacto para escenas de alta resolución"* (sección 6 de `conclusions.md`). La experiencia personal con las dos versiones de la notebook lo confirma de forma directa: fue exactamente ese cambio el que habilitó las primeras detecciones reales en SF Bay.

También coincido con que *"el etiquetado es el cuello de botella"*. Los falsos positivos que persisten después del slicing no son un problema del modelo YOLO11 en sí, sino de la distribución del dataset de entrenamiento: los chips de ShipsNet son demasiado homogéneos y no representan la diversidad visual de una escena portuaria compleja.

El recall bajo del modelo grupal (0.47 en el mejor experimento) y los falsos positivos observados en mis ejecuciones apuntan al mismo origen: **falta de diversidad en los datos de entrenamiento**, tanto en términos de negativos difíciles como de variedad de escenas y condiciones.

---

## Referencias

- SAHI: https://github.com/obss/sahi
- ShipsNet (Kaggle): https://www.kaggle.com/rhammell/ships-in-satellite-imagery
- Ultralytics YOLO11: https://docs.ultralytics.com
- Conclusiones grupales del proyecto: `conclusions.md`
