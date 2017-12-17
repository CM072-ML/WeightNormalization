# Resumen

La normalización de los pesos de una red neuronal a través de la reparametrización de vectores pesos ayuda a optimizar el tiempo de entrenamiento de una red neuronal profunda, además acelera la convergencia del **descenso de gradiente estocástico**. La reparametrización usada en la presente investigación se basa en el **batch normalization**, pero no introduce ninguna dependencia entre los ejemplos en un **minibatch**. Es decir que el presente método también puede aplicarse con éxito a modelos recurrentes como **LSTM** y a aplicaciones sensibles al ruido como el aprendizaje por reforzamiento profundo o modelos generativos, para los cuales la normalización batch es menos adecuada. La sobrecarga computacional del método es pequeña, de esta manera permite más pasos de optimización para ser usados en la misma cantidad de tiempo.

## Introducción

Los éxitos recientes en el aprendizaje profundo han demostrado que las redes neuronales entrenadas por la optimización basada en el gradiente de primer orden son capaces de lograr resultados asombrosos en diversos campos como la visión computacional, el reconocimiento del voz y el modelado del lenguaje. Sin embargo, este método depende en gran medida de la curvatura de la funcion de costo a optimizar. El radio de curvatura, y la optimización que se plantea, no es invariante a la reparameterización: puede haber múltiples formas equivalentes de parametrizar el mismo modelo. Entonces el principal objetivo sería encontrar la mejor forma de parametrizar una red neuronal.

Se han desarrollado varios métodos para mejorar el condicionamiento del gradiente de costos para arquitecturas de redes neuronales generales. Un enfoque es usar descenso de gradiente de primer orden estándar sin precondicionamiento, pero cambiar la parametrización del modelo para dar gradientes que son más parecidos a las gradientes naturales de estos métodos.

Otro enfoque en esta dirección es la normalización batch, un método en el que la salida de cada neurona (antes de la aplicación de la no linealidad) se normaliza por la media y desviación estándar de las salidas calculadas sobre los ejemplos en el minibatch. Esto reduce el desplazamiento covariable de las salidas de las neuronas.

Siguiendo este segundo enfoque para aproximar la optimización del gradiente natural, se presenta un método simple pero general, llamado normalización del peso, para mejorar la optimización de los pesos de los modelos de red neuronal. El método se inspira en la normalización batch, pero es un método determinista que no comparte la propiedad de la normalización batch de añadir ruido a los gradientes. Además, la sobrecarga impuesta por el modelo es menor: no se requiere memoria adicional y el cálculo adicional es insignificante.


## Normalización del Peso

Consideramos redes neuronales artificiales estándar donde el cálculo de cada neurona consiste en tomar una suma ponderada de características de entrada:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28w.x%20&plus;%20b%29">
</p>

donde ![](https://latex.codecogs.com/gif.latex?w) es un vector de peso k-dimensional, ![](https://latex.codecogs.com/gif.latex?b) es un término de polarización escalar, ![](https://latex.codecogs.com/gif.latex?x) es un vector k-dimensional de características de entrada, ![](https://latex.codecogs.com/gif.latex?%5Cphi%28.%29) denota una funcion no lineal elemental que denota la salida escalar de la neurona. Después de asociar una función de pérdida a una o más salidas neuronales, dicha red neuronal es comúnmente entrenada por el descenso de gradiente estocástica con los parámetros ![](https://latex.codecogs.com/gif.latex?w), ![](https://latex.codecogs.com/gif.latex?b) y además ![](https://latex.codecogs.com/gif.latex?y) denota la salida escalar de la neurona.

Con la intención de acelerar la convergencia de este procedimiento de optimización, se realizará la reparameterización de cada vector de peso ![](https://latex.codecogs.com/gif.latex?w) en términos de un vector de parámetro ![](https://latex.codecogs.com/gif.latex?v) y un parámetro escalar ![](https://latex.codecogs.com/gif.latex?g) y se calculará el descenso de gradiente estocástico con respecto a esos parámetros. La expresión del vector quedaría expresado de la siguiente forma:

<p align="center">
  <img src="https://i.imgur.com/515KiQE.png">
</p>

donde ![](https://latex.codecogs.com/gif.latex?v)  es un vector k-dimensional, ![](https://latex.codecogs.com/gif.latex?g) es un escalar, y ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%7C%7C) denota la norma euclidiana de ![](https://latex.codecogs.com/gif.latex?v). Esta reparameterización tiene el efecto de fijar la norma euclidiana del vector de peso ![](https://latex.codecogs.com/gif.latex?w), siendo ahora ![](https://latex.codecogs.com/gif.latex?%7C%7Cw%7C%7C%20%3D%20g), independiente de los parámetros ![](https://latex.codecogs.com/gif.latex?v).

## Estado del Arte

Investigaciones anteriores también desarrollaban la idea de normalizar el vector de peso, pero la optimización solo se realizaba mediante la parametrización de ![](https://latex.codecogs.com/gif.latex?w), aplicando solamente la normalización después de cada paso de descenso de gradiente estocástico. Con el presente método se reparameteriza explícitamente el modelo y realizar un descenso de gradiente estocástico en los nuevos parámetros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g) directamente. De esta forma se mejora el condicionamiento del gradiente y conduce a una convergencia mejorada del procedimiento de optimización. Al desacoplar la norma del vector de peso ![](https://latex.codecogs.com/gif.latex?g) de la dirección del vector de peso ![](https://latex.codecogs.com/gif.latex?%28v/%7C%7Cv%7C%7C%29), se acelera la convergencia de la optimización de descenso de gradiente estocástico.

## Gradientes

El entrenamiento de una red neuronal mediante la nueva parametrización se realiza utilizando métodos estándar de descenso gradiente estocástico. De esta forma se obtiene el gradiente de una función de pérdida ![](https://latex.codecogs.com/gif.latex?L) con respecto a los nuevos parámetros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g). De la siguiente forma:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_gL%20%3D%20%5Cfrac%7B%5Cnabla_wL.v%7D%7B%7C%7Cv%7C%7C%7D">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_vL%20%3D%20%5Cfrac%7Bg%7D%7B%7C%7Cv%7C%7C%7D%5Cnabla_wL%20-%20%5Cfrac%7Bg%5Cnabla_gL%7D%7B%7C%7Cv%7C%7C%5E2%7Dv">
</p>

donde ![](https://latex.codecogs.com/gif.latex?%5Cnabla_wL) es el gradiente con respecto a los pesos ![](https://latex.codecogs.com/gif.latex?w) que se usan normalmente.

Por lo tanto, la retropropagación mediante la normalización del peso sólo requiere una modificación menor de las ecuaciones habituales y se implementa fácilmente utilizando software de red neural estándar, ya sea especificando directamente la red en términos de los parámetros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g) y dependiendo de la auto-diferenciación o aplicando la ecuación anterior en una etapa posterior al procesamiento. A diferencia de la normalización batch, las expresiones anteriores son independientes del tamaño del minilote y, por tanto, causan sólo una sobrecarga computacional mínima.

Una forma alternativa de escribir el gradiente es:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_vL%20%3D%20%5Cfrac%7Bg%7D%7B%7C%7Cv%7C%7C%7DM_w%5Cnabla_wL">
</p>

con:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?M_w%20%3D%20I%20-%20%5Cfrac%7Bww%27%7D%7B%7C%7Cw%7C%7C%5E2%7D">
</p>

donde ![](https://latex.codecogs.com/gif.latex?M_w) es una matriz de proyección que se proyecta sobre el complemento del vector ![](https://latex.codecogs.com/gif.latex?w). Esto demuestra que la normalización del peso cumple dos cosas: escala el gradiente de peso por ![](https://latex.codecogs.com/gif.latex?g/%7C%7Cv%7C%7C) y proyecta el gradiente lejos del vector de peso actual. Ambos efectos ayudan a aproximar la matriz de covarianza del gradiente a la optimización de la identidad y los beneficios.

Debido a la proyección lejos de ![](https://latex.codecogs.com/gif.latex?w), la norma de ![](https://latex.codecogs.com/gif.latex?v) crece monotónicamente con las actualizaciones de los peso cuando una red neuronal aprende con la normalización de peso usando el descenso de gradiente estándar sin momento. Sea ![](https://latex.codecogs.com/gif.latex?v%27%20%3D%20v%20&plus;%20%5CDelta%20v) denotando la actualización de parámetros, con ![](https://latex.codecogs.com/gif.latex?%5CDelta%20v%20%5Cpropto%20%5Cnabla_vL) (ascenso / descenso más pronunciado), entonces ![](https://latex.codecogs.com/gif.latex?%5CDelta%20v) es necesariamente ortogonal al vector de peso actual ![](https://latex.codecogs.com/gif.latex?w), ya que nos proyectamos lejos de él al calcular ![](https://latex.codecogs.com/gif.latex?%5Cnabla_vL). Puesto que ![](https://latex.codecogs.com/gif.latex?v) es proporcional a ![](https://latex.codecogs.com/gif.latex?w), la actualización es también ortogonal a ![](https://latex.codecogs.com/gif.latex?v) y aumenta su norma mediante el teorema de Pitágoras. Específicamente, si ![](https://latex.codecogs.com/gif.latex?%7C%7C%5CDelta%20v%7C%7C/%7C%7Cv%7C%7C%20%3D%20c) el nuevo vector de peso tendrá la norma ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%27%7C%7C%20%3D%20%5Csqrt%7B%7C%7Cv%7C%7C%5E2%20&plus;%20c%5E2%7C%7Cv%7C%7C%5E2%7D%20%3D%20%5Csqrt%7B1%20&plus;%20c%5E2%7D%7C%7Cv%7C%7C%20%u2265%20%7C%7C%20v%20%7C%7C). La tasa de incremento dependerá de la varianza del gradiente de peso. Si las gradientes son ruidosas, ![](https://latex.codecogs.com/gif.latex?c) será alto y la norma de ![](https://latex.codecogs.com/gif.latex?v) aumentará rápidamente, lo que a su vez reducirá el factor de escala ![](https://latex.codecogs.com/gif.latex?g/%7C%7Cv%7C%7C). Si la norma de los gradientes es pequeña, obtendremos ![](https://latex.codecogs.com/gif.latex?%5Csqrt%7B1%20&plus;%20c%5E2%7D%20%5Cthickapprox%201), y la norma de ![](https://latex.codecogs.com/gif.latex?v) dejará de aumentar. Usando este mecanismo, el gradiente escalado se autoestabiliza su norma.

Empíricamente, la capacidad de crecer de la norma ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%7C%7C) hace que la optimización de redes neuronales con normalización de peso sea muy robusta al valor de la tasa de aprendizaje. Si la tasa de aprendizaje es demasiado grande, la norma de los pesos no normalizados crece rápidamente hasta alcanzar una tasa de aprendizaje efectiva adecuada. Una vez que la norma de los pesos ha crecido grande con respecto a la norma de las actualizaciones, la tasa de aprendizaje eficaz se estabiliza. Por lo tanto, las redes neuronales con normalización de peso funcionan bien con un rango mucho más amplio de tasas de aprendizaje que cuando se usa la parametrización normal. A su vez, las redes neuronales con normalización batch también poseen esta propiedad, y pueden ser explicado por este análisis.

## Relación con la normalización batch

Para realizar la reparameterización se usará la normalización batch, el cual normaliza las estadísticas de la preactivación ![](https://latex.codecogs.com/gif.latex?t) para cada minibatch se tiene:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%27%20%3D%20%5Cfrac%7Bt%20-%20%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D">
</p>

con ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D), ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D) la media y desviación estándar de las pre-activaciones ![](https://latex.codecogs.com/gif.latex?t%20%3D%20v%20%5Ccdot%20x). Para el caso especial en el que la red sólo tiene una sola capa, y las características de entrada ![](https://latex.codecogs.com/gif.latex?x) para esa capa son normalizadas(distribuidas independientemente con media cero y varianza unitaria), estas estadísticas son dadas por ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D%20%3D%200) y ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D%20%3D%20%7C%7Cv%7C%7C). En ese caso, la normalización de las pre-activaciones mediante normalización batch es equivalente a la normalización de los pesos mediante la normalización del peso.

Las redes neuronales convolucionales suelen tener mucho menos peso que las pre-activaciones, por lo que normalizar los pesos es a menudo mucho más barato computacionalmente. Además, la norma de v es no estocástica, mientras que la media de minibatch ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) y la varianza ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2%5Bt%5D) pueden tener en general una varianza alta para el tamaño de minibatch pequeño. Por lo tanto, la normalización del peso puede verse como una aproximación más barata y menos ruidosa a la normalización batch. Aunque la equivalencia exacta no suele mantenerse para arquitecturas más profundas, todavía se encuentra que el método de normalización de peso proporciona gran parte de la aceleración de la normalización batch completa.

## Inicialización

Además de un efecto de reparameterización, la normalización batch también tiene el beneficio de fijar la escala de las características generadas por cada capa de la red neuronal. Esto hace que la optimización sea robusta frente a las inicializaciones de parámetros para las cuales estas escalas varían entre capas. Dado que la normalización del peso carece de esta propiedad, es importante inicializar adecuadamente estos parámetros. Entonces se realizará un muestreo de los elementos de ![](https://latex.codecogs.com/gif.latex?v) mediante una distribución simple con una escala fija, por ejemplo: distribución normal con media cero y desviación estándar 0,05. Antes de iniciar el entrenamiento, se inicializa los parámetros ![](https://latex.codecogs.com/gif.latex?b) y ![](https://latex.codecogs.com/gif.latex?g) para fijar las estadísticas de minibatch de todas las pre-activaciones en la red, como en la normalización batch, pero solo para un único minibatch de datos y sólo durante la inicialización. Esto se puede realizar de manera eficiente realizando un primer paso de feedforward a través de nuestra red para un único minibatch de datos ![](https://latex.codecogs.com/gif.latex?X), usando el siguiente cálculo en cada neurona:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%20%3D%20%5Cfrac%7Bv%5Ccdot%20x%7D%7B%7C%7Cv%7C%7C%7D">
</p>

y

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28%5Cfrac%7Bt%20-%20%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D%29">
</p>

donde ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) y ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D) son la media y la desviación estándar de la preactivación ![](https://latex.codecogs.com/gif.latex?t) sobre los ejemplos en el minibatch. Entonces podemos inicializar el sesgo ![](https://latex.codecogs.com/gif.latex?b) de la neurona y la escala ![](https://latex.codecogs.com/gif.latex?g) como:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?g%20%5Cleftarrow%20%5Cfrac%7B1%7D%7B%5Csigma%5Bt%5D%7D%2C%20b%20%5Cleftarrow%20%5Cfrac%7B-%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D">
</p>

de manera que ![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28w.x%20&plus;%20b%29). Como la normalización batch, este método asegura que todas las características inicialmente tienen media cero y varianza unitaria antes de la aplicación de la no linealidad. Con el presente método esto sólo es válido para el minibatch que utilizamos para la inicialización, y los minibatches subsiguientes pueden tener estadísticas ligeramente diferentes, pero experimentalmente se encuentra que este método de inicialización funciona bien. El método también puede aplicarse a redes sin normalización de peso, simplemente haciendo una optimización de gradiente estocástico en los parámetros ![](https://latex.codecogs.com/gif.latex?w) directamente, después de la inicialización en términos de ![](https://latex.codecogs.com/gif.latex?v) y ![](https://latex.codecogs.com/gif.latex?g). La desventaja de este método de inicialización es que sólo puede aplicarse en casos similares donde la normalización de lote es aplicable. Para modelos con recursividad, como RNNs y LSTMs, tendremos que recurrir a métodos de inicialización estándar.

<p align="center">
  <img src="https://i.imgur.com/JCuSUuv.png">
</p>


## Normalización Batch

La normalización del peso, hace que la escala de las activaciones neuronales sea aproximadamente independiente de los parámetros ![](https://latex.codecogs.com/gif.latex?v). Sin embargo, a diferencia de la normalización batch, las medias de las activaciones neuronales siguen dependiendo de ![](https://latex.codecogs.com/gif.latex?v). con una versión especial de la normalización batch, a la que llamamos normalización batch only mean. Con este método de normalización, se resta las medias del minibatch como con la normalización batch completa, pero divide por las desviaciones estándar de minibatch. Es decir, calculamos las activaciones neuronales utilizando:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%20%3D%20w%5Ccdot%20x">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%27%20%3D%20t%20-%20%5Cmu%5Bt%5D%20&plus;%20b">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28t%27%29">
</p>


donde ![](https://latex.codecogs.com/gif.latex?w) es el vector de peso, parametrizado usando la normalización del peso, y ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) es la media del minibatch de la preactivación ![](https://latex.codecogs.com/gif.latex?t). Durante el entrenamiento, mantenemos un promedio de la media del minibatch que sustituimos por ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) en el tiempo de prueba.

<p align="center">
  <img src="https://i.imgur.com/aQ8w9UG.png">
</p>

El gradiente de la pérdida con respecto a la preactivación ![](https://latex.codecogs.com/gif.latex?t) se calcula como:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_tL%20%3D%20%5Cnabla_%7Bt%27%7DL%20-%20%5Cmu%5B%5Cnabla_%7Bt%27%7DL%5D">
</p>


donde ![](https://latex.codecogs.com/gif.latex?%5Cmu%5B%5Ccdot%20%5D) denota una vez más la operación para tomar la media del minibatch. Por lo tanto, la normalización batch de media sólo tiene el efecto de centrar los gradientes que son retropropagados. Esta es una operación comparativamente barata, y la sobrecarga computacional de la normalización batch de sólo media es por lo tanto menor que para la normalización batch completa. Además, este método produce menos ruido durante el entrenamiento, y el ruido que se produce es más suave, ya que la ley de grandes números asegura que ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%20%5D) y ![](https://latex.codecogs.com/gif.latex?%5Cmu%5B%5Cnabla%20t%5D) están aproximadamente distribuidos normalmente.


## DRAW: Una red neuronal recurrente para la generación de imágenes

Este artículo muestra la arquitecture de red neuronal DRAW(Deep Recurrent Attentative Writer) para la generación de imágenes. DRAW imita los ojos humanos especialmente el mecanismo de atención espacial, de esta forma con una variación secuencial autoencoding que hace posible la construcción iterativa de imágenes complejas.

<p align="center">
  <img src="https://i.imgur.com/a/kUEXU">
</p>

### Introducción 

La arquitectura DRAW(Deep Recurrent Attentative Writer) representa un cambio hacia una forma mas natural de construcción de imágenes, en que parte de una escena es creada independiente de otra.

La arquitetura DRAW se basa principalmente en un par de redes neuronales recurrentes: red **encoder**  que comprime las imágenes reales presentadas durante el entrenamiento y una red **decoder** que reconstituye imágenes después de recibir códigos.

### La red DRAW

Encoder y decoder son redes recurrentes en DRAW, la secuencia de muestras de código es intercambiada entre ellos, el encoder es privado para la salida del decoder anterior. La salida del decoder es exitosamente agregado a la distribución que generará los datos. Una mecanismo de distracción que se actualiza dinámicamente es usada para restringir la región de entrada observada por el encoder y la región de salida modificada por el decoder. La red decide en cada iteración "donde leer" y "donde escribir" tanto como "qué escribir"

#### Arquitectura de Red

En general el encoder y decoder podria ser implementado por cualquier red neuronal recurrente. Pero en los experimentos se usa la arquitectura Long Short-Term Memory (LSTM) para ambos.





