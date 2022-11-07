 <p align="left">
   <img src="https://img.shields.io/badge/status-en%20desarrollo-green"> 
   <img src="https://img.shields.io/github/issues/jesusdanielquiroga/model-api">
   <img src="https://img.shields.io/github/forks/jesusdanielquiroga/model-api">
   <img src="https://img.shields.io/github/forks/jesusdanielquiroga/model-api">
   <img src="https://img.shields.io/github/license/jesusdanielquiroga/model-api">
   <a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fjesusdanielquiroga%2FSeries-de-Tiempo"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fjdquiroga2410"></a>
   <img src="https://img.shields.io/github/stars/camilafernanda?style=social">
   <img src="https://img.shields.io/badge/topic-seriesdetiempo-orange">
  </p>
  
<img width="930" alt="intro pred" src="https://user-images.githubusercontent.com/87950040/200219442-76a09312-b177-4b89-b5d5-2203f7e31a1d.png">

<h1> Pronóstico de series de tiempo </h1>

## Índice

* [Índice](#índice)

* [Intro](#intro)

* [Entrenamiento de un modelo de forecasting](#entrenamiento-de-un-modelo-de-forecasting)

* [Predicciones multi-step](#predicciones-multi-step)

* [Recursive multi-step forecasting](##recursive-multi-step-forecasting)

* [Direct multi-step forecasting](##direct-multi-step-forecasting)

* [Multiple output forecasting](##multiple-output-forecasting)

* [Forecasting autorregresivo recursivo](#forecasting-autorregresivo-recursivo)

# Intro

El pronóstico de <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">series temporales</a> es un área importante del aprendizaje automático que a menudo se descuida. Es importante porque hay tantos problemas de predicción que involucran un componente de tiempo. Estos problemas se descuidan porque es este componente de tiempo el que hace que los problemas de <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">series temporales</a> sean más difíciles de manejar.

Hacer predicciones sobre el futuro se llama extrapolación en el manejo estadístico clásico de datos de series de tiempo. El pronóstico implica tomar modelos que se ajusten a los datos históricos y usarlos para predecir observaciones futuras, siempre teiendo en cuenta que el futuro no está disponible y solo debe estimarse a partir de lo que ya sucedió.

Al pronosticar, es importante comprender su objetivo.

Utilice el método socrático y haga muchas preguntas para ayudar a ampliar los detalles de su problema de modelado predictivo . Por ejemplo:

* ¿Cuántos datos tienes disponibles y eres capaz de recopilarlos todos juntos? Más datos a menudo son más útiles, ya que ofrecen una mayor oportunidad para el análisis exploratorio de datos, la prueba y el ajuste del modelo y la fidelidad del modelo.
* ¿Cuál es el horizonte temporal de las predicciones que se requiere? ¿Corto, mediano o largo plazo? Los horizontes de tiempo más cortos suelen ser más fáciles de predecir con mayor confianza.
* ¿Se pueden actualizar los pronósticos con frecuencia a lo largo del tiempo o se deben hacer una vez y permanecer estáticos? Actualizar los pronósticos a medida que se dispone de nueva información a menudo da como resultado predicciones más precisas.
* ¿Con qué frecuencia temporal se requieren los pronósticos? A menudo, los pronósticos se pueden hacer a frecuencias más bajas o más altas, lo que le permite aprovechar el muestreo descendente y el muestreo ascendente de datos, lo que a su vez puede ofrecer beneficios durante el modelado.

En el siguiente repositorio se describe cómo utilizar modelos de regresión de Scikit-learn para realizar forecasting sobre <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">series temporales</a>. Se hace uso de <a href="https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html">Skforecast</a>, una librería que contiene las clases y funciones necesarias para adaptar cualquier modelo de regresión de Scikit-learn a problemas de forecasting.

# Entrenamiento de un modelo de forecasting

Lo primero que debemos hacer es transformar la <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">serie temporal</a> en un matriz en la que, cada valor, está asociado a la ventana temporal (lags) que le precede.

![transform_timeseries](https://user-images.githubusercontent.com/87950040/200323192-c64b6130-595e-43cd-b481-ae722a2481e8.gif)

Este tipo de transformación también permite incluir variables exógenas a la <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">serie temporal</a>.

![matrix_transformation_with_exog_variable](https://user-images.githubusercontent.com/87950040/200323505-c44bbf42-fe8a-4bc2-aa24-a3450cfec83b.png)

# Predicciones multi-step

Cuando se trabaja con <a href="https://github.com/jesusdanielquiroga/Series-de-Tiempo.git">series temporales</a>, raramente se quiere predecir solo el siguiente elemento de la serie ( $t_{+1}$ ), sino todo un intervalo futuro o un punto alejado en el tiempo ( $t_{+n}$ ). A cada paso de predicción se le conoce como step. Existen varias estrategias que permiten generar este tipo de predicciones múltiples.

## Recursive multi-step forecasting

Dado que, para predecir el momento  tn  se necesita el valor de  $t_{n−1}$ , y  $t_{n−1}$  se desconoce, se sigue un proceso recursivo en el que, cada nueva predicción, hace uso de la predicción anterior. A este proceso se le conoce como recursive forecasting o recursive multi-step forecasting y pueden generarse fácilmente con las clases $ForecasterAutoreg$ y $ForecasterAutoregCustom$ de la librería <a href="https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html">Skforecast</a>.

![diagrama-multistep-recursiva](https://user-images.githubusercontent.com/87950040/200326635-d2c7b1ab-25b9-4945-b465-0502d4497813.png)

## Direct multi-step forecasting

El método direct multi-step forecasting consiste en entrenar un modelo distinto para cada step. Por ejemplo, si se quieren predecir los siguientes 5 valores de una serie temporal, se entrenan 5 modelos distintos, uno para cada step. Como resultado, las predicciones son independientes unas de otras.

![diagrama-prediccion-multistep-directa](https://user-images.githubusercontent.com/87950040/200329066-cb451d71-f9cf-4870-b52e-c9b6ec3b58d1.png)

La principal complejidad de esta aproximación consiste en generar correctamente las matrices de entrenamiento para cada modelo. Todo este proceso está automatizado en la clase $ForecasterAutoregDirect$ de la librería <a href="https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html">Skforecast</a>. También es importante tener en cuenta que esta estrategia tiene un coste computacional más elevado ya que requiere entrenar múltiples modelos. En el siguiente esquema se muestra el proceso para un caso en el que se dispone de la variable respuesta y dos variables exógenas.

![diagram_skforecast_multioutput](https://user-images.githubusercontent.com/87950040/200329390-1a00b922-a0bb-44c4-8bfd-49daaecfdcc0.png)

## Multiple output forecasting

Determinados modelos, por ejemplo, las redes neuronales LSTM, son capaces de predecir de forma simultánea varios valores de una secuencia (one-shot).

# Forecasting autorregresivo recursivo

<a href="Forecasting_series_temporales.ipynb">**Ejercicio 1**</a>

Se dispone de una serie temporal con el gasto mensual (millones de dólares) en fármacos con corticoides que tuvo el sistema de salud Australiano entre 1991 y 2008. Se pretende crear un modelo autoregresivo capaz de predecir el futuro gasto mensual.

Es posible que se requiera instalar en su entorno la librería <a href="https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html">Skforecast</a>:

```sh
!pip install skforecast
```
Las librerías requeridas para el ejercicio:

```sh
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
%matplotlib inline

# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# Configuración warnings
# ==============================================================================
import warnings
# warnings.filterwarnings('ignore')
```

**Datos**

```sh
# Descarga de datos
# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
datos = pd.read_csv(url, sep=',')
```

**Preparación de datos**

La columna fecha se ha almacenado como string. Para convertirla en datetime, se emplea la función pd.to_datetime(). Una vez en formato datetime, y para hacer uso de las funcionalidades de Pandas, se establece como índice. Además, dado que los datos son mensuales, se indica la frecuencia (Monthly Started 'MS').
```sh
# Preparación del dato
# ==============================================================================
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y/%m/%d')
datos = datos.set_index('fecha')
datos = datos.rename(columns={'x': 'y'})
datos = datos.asfreq('MS')
datos = datos.sort_index()
datos.head()
```
Al establecer una frecuencia con el método $asfreq()$, Pandas completa los huecos que puedan existir en la serie temporal con el valor de $Null$ con el fin de asegurar la frecuencia indicada. Por ello, se debe comprobar si han aparecido missing values tras esta transformación.

**Dividir datos de train y test**

```sh
# Separación datos train-test
# ==============================================================================
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
ax.legend();
```
![figure1 (1)](https://user-images.githubusercontent.com/87950040/200336777-d6efd07b-0f78-4c8a-b760-ffdf905f8998.png)

**ForecasterAutoreg**

Se crea y entrena un modelo ForecasterAutoreg a partir de un regresor RandomForestRegressor y una ventana temporal de 6 lags. Esto último significa que, el modelo, utiliza como predictores los 6 meses anteriores.
```sh
# Crear y entrenar forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 6
             )

forecaster.fit(y=datos_train['y'])
forecaster
```
```sh
================= 
ForecasterAutoreg 
================= 
Regressor: RandomForestRegressor(random_state=123) 
Lags: [1 2 3 4 5 6] 
Transformer for y: None 
Transformer for exog: None 
Window size: 6 
Included exogenous: False 
Type of exogenous variable: None 
Exogenous variables names: None 
Training range: [Timestamp('1992-04-01 00:00:00'), Timestamp('2005-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 123, 'verbose': 0, 'warm_start': False} 
Creation date: 2022-11-07 14:33:03 
Last fit date: 2022-11-07 14:33:03 
Skforecast version: 0.5.1 
Python version: 3.7.15 
```
**Predicciones**
```sh
# Predicciones
# ==============================================================================
steps = 36
predicciones = forecaster.predict(steps=steps)
predicciones.head(5)
```
```sh
2005-07-01    0.878756
2005-08-01    0.882167
2005-09-01    0.973184
2005-10-01    0.983678
2005-11-01    0.849494
Freq: MS, Name: pred, dtype: float64
```
**Graficamos resultados**
```sh
# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
```
![figure1 (2)](https://user-images.githubusercontent.com/87950040/200339050-62ad75a3-8df4-4c2c-9810-e9d735bef74a.png)

**Estimación del error**

Se cuantifica el error que comete el modelo en sus predicciones. En este caso, se emplea como métrica el mean squared error (mse)
```sh
# Error test
# ==============================================================================
error_mse = mean_squared_error(
                y_true = datos_test['y'],
                y_pred = predicciones
            )

print(f"Error de test (mse): {error_mse}")
```
```sh
Error de test (mse): 0.07326833976120374
```
**Ajuste de hiperparámetros (tuning)**

El **$ForecasterAutoreg$** entrenado ha utilizado una ventana temporal de 6 lags y un modelo Random Forest con los hiperparámetros por defecto. Sin embargo, no hay ninguna razón por la que estos valores sean los más adecuados. Para identificar la mejor combinación de lags e hiperparámetros, la librería Skforecast dispone de la función **$grid_search_forecaster$** con la que comparar los resultados obtenidos con cada configuración del modelo.

```sh
# Grid search de hiperparámetros
# ==============================================================================
steps = 36
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # Este valor será remplazado en el grid search
             )

# Lags utilizados como predictores
lags_grid = [10, 20]

# Hiperparámetros del regresor
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train['y'],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
                  )             
 ```    
  ```sh
  Number of models compared: 12.
loop lags_grid:   0%|                                               | 0/2 [00:00<?, ?it/s]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:01<00:05,  1.15s/it]
loop param_grid:  33%|████████████▋                         | 2/6 [00:06<00:14,  3.64s/it]
loop param_grid:  50%|███████████████████                   | 3/6 [00:07<00:07,  2.51s/it]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:13<00:07,  3.70s/it]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:14<00:02,  2.81s/it]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:20<00:00,  3.83s/it]
loop lags_grid:  50%|███████████████████▌                   | 1/2 [00:20<00:20, 20.28s/it]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:01<00:06,  1.26s/it]
loop param_grid:  33%|████████████▋                         | 2/6 [00:06<00:15,  3.87s/it]
loop param_grid:  50%|███████████████████                   | 3/6 [00:08<00:08,  2.68s/it]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:14<00:07,  3.92s/it]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:15<00:02,  2.97s/it]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:21<00:00,  4.09s/it]
loop lags_grid: 100%|███████████████████████████████████████| 2/2 [00:41<00:00, 20.95s/it]
`Forecaster` refitted using the best-found lags and parameters, and the whole data set: 
  Lags: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] 
  Parameters: {'max_depth': 3, 'n_estimators': 500}
  Backtesting metric: 0.012836389345193383
  ```  
  ```sh
# Resultados Grid Search
# ==============================================================================
resultados_grid
  ``` 
 ```sh
                                              	lags                               	params	mean_squared_errormax_depthn_estimators
7	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 3, 'n_estimators': 500}	0.012836	3	500
6	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 3, 'n_estimators': 100}	0.012858	3	100
9	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 5, 'n_estimators': 500}	0.013248	5	500
8	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 5, 'n_estimators': 100}	0.013364	5	100
11	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 10, 'n_estimators': 500}	0.013435	10	500
10	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...	{'max_depth': 10, 'n_estimators': 100}	0.014028	10	100
0	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 3, 'n_estimators': 100}	0.036982	3	100
1	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 3, 'n_estimators': 500}	0.037345	3	500
3	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 5, 'n_estimators': 500}	0.037574	5	500
5	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 10, 'n_estimators': 500}	0.040542	10	500
2	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 5, 'n_estimators': 100}	0.041474	5	100
4	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	{'max_depth': 10, 'n_estimators': 100}	0.047870	10	100
 ``` 
Los mejores resultados se obtienen si se utiliza una ventana temporal de 20 lags y una configuración de Random Forest {'max_depth': 3, 'n_estimators': 500}.
 
**Modelo Final**
 
Finalmente, se entrena de nuevo un ForecasterAutoreg con la configuración óptima encontrada mediante validación. Este paso no es necesario si se indica return_best = True en la función grid_search_forecaster.

```sh
# Crear y entrenar forecaster con mejores hiperparámetros
# ==============================================================================
regressor = RandomForestRegressor(max_depth=3, n_estimators=500, random_state=123)
forecaster = ForecasterAutoreg(
                regressor = regressor,
                lags      = 20
             )

forecaster.fit(y=datos_train['y'])
```
```sh
# Predicciones
# ==============================================================================
predicciones = forecaster.predict(steps=steps)
```
```sh
# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
```
![figure2](https://user-images.githubusercontent.com/87950040/200343122-26c2fdea-cde6-4aad-b59c-aebc81930c7d.png)

```sh
# Error de test
# ==============================================================================
error_mse = mean_squared_error(
                y_true = datos_test['y'],
                y_pred = predicciones
            )

print(f"Error de test (mse) {error_mse}")
```
```sh
Error de test (mse) 0.004392699665157793
```
