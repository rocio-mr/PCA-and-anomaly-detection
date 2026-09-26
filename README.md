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

