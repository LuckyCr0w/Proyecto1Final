# Análisis de Machine Learning - Retail Sales Dataset

## Felipe Lucciano Santino Di Vanni Valenzuela

---

## **Descripción del Proyecto**

Este proyecto implementa un pipeline completo de machine learning para análisis de datos de ventas retail, dividido en tres notebooks modulares y conectados entre sí.

## **Estructura del Proyecto**

```
retailAnalisis/
├── data/
│   ├── retail_sales_dataset.csv          # Dataset original
│   ├── retail_eda_processed.csv           # Datos procesados del EDA
│   ├── eda_summary.pkl                    # Resumen de hallazgos del EDA
│   └── datos_ml_procesados.pkl            # Datos preparados para ML
├── notebook/
│   ├── EDA.ipynb                          # Análisis Exploratorio de Datos
│   ├── Preprocessing.ipynb                # Preprocesamiento de datos
│   └── Benchmarking.ipynb                 # Comparación de modelos
├── reports/
│   ├── benchmarking_results.pkl           # Resultados del benchmarking
│   └── best_model.pkl                     # Mejor modelo entrenado
├── presentation/
└── README.md                              # Este archivo
```

---

## **Pipeline de Machine Learning**

### **1. EDA.ipynb - Análisis Exploratorio de Datos**
**Objetivo:** Comprensión profunda del dataset y identificación de patrones

**Contenido:**
- Carga y exploración inicial del dataset
- Análisis univariado de variables numéricas y categóricas
- Análisis bivariado y matriz de correlaciones
- Análisis temporal de patrones de ventas
- Detección de outliers y valores atípicos
- Síntesis de hallazgos e insights de negocio

**Output:** 
- `retail_eda_processed.csv`: Dataset con transformaciones básicas
- `eda_summary.pkl`: Resumen de hallazgos y recomendaciones

### **2. Preprocessing.ipynb - Preprocesamiento de Datos**
**Objetivo:** Preparación de datos para machine learning

**Contenido:**
- Carga de datos del EDA
- Definición del problema de clasificación
- Encoding de variables categóricas
- Escalado de features numéricas
- División estratificada de datos (train/test)
- Validación de transformaciones

**Input:** `retail_eda_processed.csv`, `eda_summary.pkl`
**Output:** `datos_ml_procesados.pkl`

### **3. Benchmarking.ipynb - Comparación de Modelos**
**Objetivo:** Implementación y evaluación de algoritmos de ML

**Contenido:**
- Implementación de 4 modelos: LightGBM, Random Forest, Decision Tree, KNN
- Entrenamiento y evaluación comparativa
- Métricas de performance: Accuracy, Precision, Recall, F1-Score
- Visualizaciones de resultados y matrices de confusión
- Selección del mejor modelo
- Generación de reportes finales

**Input:** `datos_ml_procesados.pkl`
**Output:** `benchmarking_results.pkl`, `best_model.pkl`

---

## **Cómo Ejecutar el Proyecto**

### **Prerrequisitos**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib
```

### **Ejecución Secuencial**
Los notebooks deben ejecutarse en orden:

1. **Ejecutar EDA.ipynb**
   ```bash
   jupyter notebook notebook/EDA.ipynb
   ```
   - Genera archivos necesarios para el siguiente paso
   - Proporciona insights del negocio

2. **Ejecutar Preprocessing.ipynb**
   ```bash
   jupyter notebook notebook/Preprocessing.ipynb
   ```
   - Usa outputs del EDA
   - Prepara datos para modelado

3. **Ejecutar Benchmarking.ipynb**
   ```bash
   jupyter notebook notebook/Benchmarking.ipynb
   ```
   - Usa datos procesados
   - Genera modelo final y reportes

---

## **Características del Análisis**

### **Dataset**
- **Fuente:** Retail Sales Dataset (Kaggle)
- **Registros:** 1,000 transacciones
- **Período:** 1 año completo
- **Variables:** Cliente, producto, temporales, monetarias

### **Modelos Implementados**
1. **LightGBM:** Gradient boosting optimizado
2. **Random Forest:** Ensemble de árboles de decisión
3. **Decision Tree:** Árbol individual para interpretabilidad
4. **K-Nearest Neighbors:** Clasificación por proximidad

### **Problema de ML**
- **Tipo:** Clasificación multiclase
- **Target:** Categorías de ventas (Alto, Medio, Bajo)
- **Features:** Variables demográficas, de producto y temporales

---

## **Metodología y Estándares**

### **Convenciones de Código**
- Variables en **camelCase** (ej: `featuresNumericas`, `dfResultados`)
- Sin uso de iconos en comentarios
- Comentarios explicativos en todas las funciones
- Explicaciones detalladas en markdown

### **Reproducibilidad**
- Semillas aleatorias fijas (`random_state=42`)
- Versiones específicas de librerías
- Documentación completa de transformaciones

### **Calidad del Código**
- Validación de datos en cada etapa
- Manejo de errores y excepciones
- Guardado automático de resultados intermedios
- Logs de progreso y verificación

---

## **Resultados y Insights**

### **Hallazgos del EDA**
- Dataset limpio sin valores faltantes
- Distribuciones balanceadas entre categorías
- Patrones temporales identificados
- Correlaciones relevantes para modelado

### **Performance de Modelos**
- Todos los modelos superan baseline de 60% accuracy
- Modelos ensemble muestran mejor performance
- Features temporales y demográficas son importantes
- Balance adecuado entre interpretabilidad y performance

### **Aplicaciones de Negocio**
- Segmentación automática de clientes
- Predicción de categorías de ventas
- Optimización de estrategias de marketing
- Planificación de inventario basada en patrones

---

## **Próximos Pasos**

1. **Optimización de Hiperparámetros:** Grid search para mejores parámetros
2. **Feature Engineering:** Creación de variables adicionales
3. **Validación Temporal:** Evaluación con datos más recientes
4. **Deployment:** Implementación en producción
5. **Monitoreo:** Seguimiento de performance en tiempo real

---

## **Contacto**

**Autor:** Felipe Lucciano Santino Di Vanni Valenzuela  
**Proyecto:** Análisis de Machine Learning - Retail Sales Dataset  
**Fecha:** 2024

---

## **Licencia**

Este proyecto es de uso educativo y de investigación. 