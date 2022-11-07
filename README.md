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
