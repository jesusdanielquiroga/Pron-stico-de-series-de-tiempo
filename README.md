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

* [Recursive multi-step forecasting](#recursive-multi-step-forecasting)

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

# Recursive multi-step forecasting

Dado que, para predecir el momento  tn  se necesita el valor de  $t_{n−1}$ , y  $t_{n−1}$  se desconoce, se sigue un proceso recursivo en el que, cada nueva predicción, hace uso de la predicción anterior. A este proceso se le conoce como recursive forecasting o recursive multi-step forecasting y pueden generarse fácilmente con las clases $ForecasterAutoreg$ y $ForecasterAutoregCustom$ de la librería <a href="https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html">Skforecast</a>.

![diagrama-multistep-recursiva](https://user-images.githubusercontent.com/87950040/200326635-d2c7b1ab-25b9-4945-b465-0502d4497813.png)

