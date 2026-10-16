# Semana 8 — Implementación y comparación de algoritmos de clasificación

**CADI: Introducción a Machine Learning | Universidad de Cundinamarca**

## Descripción

Este proyecto implementa y compara cinco algoritmos de clasificación sobre el conjunto de datos **Breast Cancer Wisconsin Diagnostic**, usando la misma partición de entrenamiento y prueba para todos los modelos. Se evalúan las métricas **Accuracy, Precision, Recall, F1-score** y se analizan las **matrices de confusión**, con especial énfasis en los falsos negativos de la clase maligna.

> **Alcance académico:** este ejercicio no constituye una herramienta clínica ni debe utilizarse para diagnosticar o tomar decisiones médicas.

## Dataset

- **Fuente:** `sklearn.datasets.load_breast_cancer`
- **Observaciones:** 569
- **Variables de entrada:** 30 características numéricas
- **Variable objetivo:** diagnóstico binario
  - `0` = Benigno
  - `1` = Maligno
- **Distribución de clases:**
  - Benigno: 357
  - Maligno: 212
- **Valores faltantes:** 0

## Modelos evaluados

| Modelo | Configuración principal |
|---|---|
| Decision Tree | `max_depth=4`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| SVM | Kernel RBF, `C=1.0`, `gamma='scale'`, `StandardScaler` |
| K-NN | `n_neighbors=5`, `StandardScaler` |
| Naive Bayes | `GaussianNB` |

## Metodología

- División entrenamiento/prueba: **80/20**
- Estratificación por clase: `stratify=y`
- Semilla de reproducibilidad: `random_state=42`
- Entrenamiento: 455 observaciones
- Prueba: 114 observaciones
- Proporción de malignos:
  - Entrenamiento: 0.374
  - Prueba: 0.368
- Métricas calculadas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Matriz de confusión
  - Falsos negativos (maligno predicho como benigno)

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-score | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 | 3 |
| SVM | 0.974 | 1.000 | 0.929 | 0.963 | 3 |
| K-NN | 0.956 | 0.974 | 0.905 | 0.938 | 4 |
| Naive Bayes | 0.939 | 1.000 | 0.833 | 0.909 | 7 |
| Decision Tree | 0.912 | 0.944 | 0.810 | 0.872 | 8 |

## Análisis de resultados

- **Random Forest** y **SVM** obtienen el mejor desempeño conjunto: Accuracy de 0.974, Precision de 1.000, Recall de 0.929 y F1-score de 0.963.
- Ambos modelos identifican **39 de los 42 casos malignos** de prueba y dejan **3 falsos negativos**.
- **K-NN** muestra un rendimiento intermedio, con 4 falsos negativos.
- **Naive Bayes** obtiene una Precision perfecta, pero su Recall es más bajo (0.833), con 7 falsos negativos.
- **Decision Tree** presenta el rendimiento más bajo en Recall (0.810) y la mayor cantidad de falsos negativos (8).

## Selección del modelo

Para esta ejecución, **Random Forest** y **SVM** alcanzan las mejores métricas y comparten el mayor Recall. Random Forest es una elección razonable por su desempeño y porque permite inspeccionar la importancia de las características. SVM obtiene las mismas métricas en esta división.

Esta selección es **académica**. Para decidir entre modelos en un contexto real se debería considerar:

- Validación cruzada
- Ajuste de umbrales de decisión
- Costos asociados a cada tipo de error
- Evaluación externa con nuevos datos

## Estructura del notebook

1. Importación de librerías
2. Carga del conjunto de datos
3. Exploración inicial
4. Variables de entrada y variable objetivo
5. División entrenamiento/prueba
6. Configuración de los cinco clasificadores
7. Función común de evaluación
8. Decision Tree — Árbol de decisión
9. Random Forest
10. Support Vector Machine (SVM)
11. K-Nearest Neighbors (K-NN)
12. Naive Bayes (GaussianNB)
13. Comparación gráfica de métricas
14. Tabla comparativa y análisis de errores
15. Selección del modelo

## Requisitos

- Python 3.13.5 o compatible
- pandas
- numpy
- matplotlib
- scikit-learn

Instalación:

```bash
pip install pandas numpy matplotlib scikit-learn
