# PCA and Anomaly Detection - Activity 9

## PARTE I:

## 1. Reconocimiento de rostros con Eigenfaces
## 📌 Descripción

En este proyecto se utiliza el Olivetti Faces Dataset disponible en scikit-learn para desarrollar un sistema de reconocimiento de rostros basado en Eigenfaces.

El objetivo es reconocer a qué persona pertenece una imagen facial utilizando técnicas clásicas de Machine Learning, sin utilizar redes neuronales.

La técnica principal empleada es PCA (Principal Component Analysis), mediante la cual se reduce la dimensionalidad de las imágenes y se obtienen las denominadas Eigenfaces.

## 📊 Dataset

Se utiliza el dataset Olivetti Faces, disponible mediante scikit-learn.

El conjunto de datos contiene:

400 imágenes
40 personas
10 imágenes por persona
Imágenes en escala de grises
Resolución de 64 × 64 píxeles

Cada imagen puede representarse como un vector de:

64 × 64 = 4096 características

Por lo tanto, cada rostro inicialmente se representa mediante 4096 valores de intensidad de píxel.

## 🧠 PCA y Eigenfaces

Se utiliza Principal Component Analysis (PCA) para reducir la dimensionalidad de las imágenes.

PCA encuentra nuevas direcciones que explican la mayor cantidad posible de variabilidad de los datos.

En el contexto del reconocimiento facial, los componentes principales obtenidos por PCA reciben el nombre de Eigenfaces.

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=100,
    whiten=True,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```
Cada componente de PCA puede visualizarse nuevamente como una imagen de 64 × 64 píxeles:

``` python
eigenface = pca.components_[0].reshape(64, 64)
```

De esta manera es posible visualizar las características principales aprendidas por PCA.

## 👤 Reconocimiento facial

Después de reducir la dimensionalidad mediante PCA, se utiliza un clasificador K-Nearest Neighbors (KNN) para reconocer la identidad de una persona.

El proceso es:

```text
 Imagen facial
      ↓
Vectorización
      ↓
PCA
      ↓
Espacio de Eigenfaces
      ↓
KNN
      ↓
Persona reconocida
```

KNN compara la representación de una nueva imagen con las imágenes del conjunto de entrenamiento y determina la clase a partir de sus vecinos más cercanos.

## 📈 Evaluación

El conjunto de datos se divide en entrenamiento (**train**) y prueba (**test**). Posteriormente se evalúa el reconocimiento usando métricas como: 

* Accuracy
* Matriz de confusión
* Classification Report

El uso de PCA permite trabajar con una representación considerablemente menor que los 4096 píxeles originales.

---

## PARTE 2: SISTEMA DE RECOMENDACIÓN DE ARTÍCULOS

## 📌 Descripción

En este proyecto se plantea el problema de un periódico en línea que desea recomendar artículos similares al artículo que un usuario está leyendo actualmente.

El objetivo es encontrar documentos con contenido temáticamente similar utilizando técnicas de procesamiento de lenguaje natural y aprendizaje no supervisado.

Para ello se utilizan:

TF-IDF
NMF
Similitud del coseno

No se utilizan redes neuronales.

📊 Dataset

Se utiliza el dataset 20 Newsgroups, disponible en scikit-learn.

Este dataset contiene documentos pertenecientes a diferentes categorías temáticas.

Algunas categorías son:

* comp.graphics
* sci.space
* rec.sport.baseball
* rec.autos
* talk.politics
* sci.med

Las categorías originales sirven como referencia para conocer el tema al que pertenece cada documento.

## 📝 Representación TF-IDF

Los documentos se convierten en una representación numérica mediante TF-IDF (Term Frequency-Inverse Document Frequency).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(newsgroups.data)
```

TF-IDF permite asignar mayor importancia a las palabras que son relevantes para determinados documentos y menor importancia a aquellas que aparecen frecuentemente en muchos documentos.

## 🔍 NMF

Se utiliza Non-Negative Matrix Factorization (NMF) para obtener una representación de los documentos basada en temas latentes.


```python
from sklearn.decomposition import NMF

nmf = NMF(
    n_components=20,
    random_state=42
)

W = nmf.fit_transform(X_tfidf)
H = nmf.components_
```

Los componentes de NMF pueden interpretarse como temas. Y de esta manera, se visualizará además las palabras principales de cada tema. Esto permitirá interpretar qué palabras caracterizan cada uno de los temas descubiertos.

## 📐 Similitud del coseno

Una vez obtenida la representación NMF de cada artículo, se utiliza la similitud del coseno para determinar qué documentos son más similares.

```python
from sklearn.metrics.pairwise import cosine_similarity

similaridades = cosine_similarity(
    W,
    W
)
```

Para un artículo seleccionado se obtienen los documentos con mayor similitud.

```text
Artículo seleccionado
        ↓
Representación NMF
        ↓
Similitud del coseno
        ↓
Artículos más similares
        ↓
Recomendaciones
```

## 🎯 Resultado

El sistema permite seleccionar un artículo y obtener otros documentos que presentan una representación temática similar.

Este enfoque puede utilizarse como base para un sistema de recomendación de contenido en un periódico digital.

--- 

## PARTE III: DETECCIÓN DE ANOMALÍAS EN SERIES TEMPORALES

## 📌 Descripción

En este proyecto se desarrolla un modelo ensemble para detectar anomalías en series temporales.

Una anomalía es una observación cuyo comportamiento se encuentra significativamente alejado del patrón habitual de la serie.

Para abordar el problema se utilizan tres algoritmos:

* Isolation Forest
* Local Outlier Factor
* One-Class SVM

Posteriormente se combinan sus resultados mediante votación mayoritaria.

## 📊 Dataset

Se utiliza una serie temporal perteneciente al Numenta Anomaly Benchmark (NAB).

El dataset contiene series temporales orientadas al estudio y evaluación de métodos de detección de anomalías.

La serie utilizada contiene principalmente:

*timestamp*
*value*

Donde timestamp representa el instante de la medición y value el valor registrado.

## ⚙️ Ingeniería de características

A partir de la variable value se generan características que representan el comportamiento temporal.

``` python
Valor anterior
df["value_lag1"] = df["value"].shift(1)
Segundo valor anterior
df["value_lag2"] = df["value"].shift(2)
Media móvil
df["rolling_mean"] = (
    df["value"]
    .rolling(window=10)
    .mean()
)
Desviación estándar móvil
df["rolling_std"] = (
    df["value"]
    .rolling(window=10)
    .std()
)
Diferencia entre valores consecutivos
df["difference"] = (
    df["value"] -
    df["value_lag1"]
)

```

Finalmente:

``` python
df = df.dropna().reset_index(drop=True)
```

## 📐 Matriz de características

Las características generadas se utilizan para construir X:

```python
X = df[
    [
        "value",
        "value_lag1",
        "value_lag2",
        "rolling_mean",
        "rolling_std",
        "difference"
    ]
]
```
Por lo tanto, cada observación está representada mediante información sobre su valor actual, valores anteriores y comportamiento reciente.

## 🌲 Isolation Forest

Isolation Forest es un algoritmo basado en árboles que busca identificar observaciones que pueden aislarse fácilmente del resto de los datos.

## 📍 Local Outlier Factor

LOF detecta observaciones que presentan una densidad local diferente a la de sus vecinos.

## 🟣 One-Class SVM

One-Class SVM aprende el comportamiento considerado normal y detecta observaciones que se encuentran fuera de esa región.

🤝 Ensemble

Las predicciones de los tres modelos se combinan:

```python
df["votos"] = (
    df["IF"] +
    df["LOF"] +
    df["SVM"]
)
``` 

Se considera una observación como anomalía cuando al menos dos modelos coinciden:

```python
df["anomalia"] = (
    df["votos"] >= 2
).astype(int)
```
Por lo tanto:

``` text
Isolation Forest ──┐
                   │
LOF ───────────────┼──→ Votación mayoritaria → Anomalía
                   │
One-Class SVM ─────┘
```

Este enfoque permite combinar diferentes estrategias de detección y obtener una decisión conjunta

## 👩‍💻 Autoría realizada por:

**Milesa Rocio Maquera Ramos**
