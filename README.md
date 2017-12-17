# Resumen

La normalizaciÃ³n de los pesos de una red neuronal a travÃ©s de la reparametrizaciÃ³n de vectores pesos ayuda a optimizar el tiempo de entrenamiento de una red neuronal profunda, ademÃ¡s acelera la convergencia del **descenso de gradiente estocÃ¡stico**. La reparametrizaciÃ³n usada en la presente investigaciÃ³n se basa en el **batch normalization**, pero no introduce ninguna dependencia entre los ejemplos en un **minibatch**. Es decir que el presente mÃ©todo tambiÃ©n puede aplicarse con Ã©xito a modelos recurrentes como **LSTM** y a aplicaciones sensibles al ruido como el aprendizaje por reforzamiento profundo o modelos generativos, para los cuales la normalizaciÃ³n batch es menos adecuada. La sobrecarga computacional del mÃ©todo es pequeÃ±a, de esta manera permite mÃ¡s pasos de optimizaciÃ³n para ser usados en la misma cantidad de tiempo.

## IntroducciÃ³n

Los Ã©xitos recientes en el aprendizaje profundo han demostrado que las redes neuronales entrenadas por la optimizaciÃ³n basada en el gradiente de primer orden son capaces de lograr resultados asombrosos en diversos campos como la visiÃ³n computacional, el reconocimiento del voz y el modelado del lenguaje. Sin embargo, este mÃ©todo depende en gran medida de la curvatura de la funcion de costo a optimizar. El radio de curvatura, y la optimizaciÃ³n que se plantea, no es invariante a la reparameterizaciÃ³n: puede haber mÃºltiples formas equivalentes de parametrizar el mismo modelo. Entonces el principal objetivo serÃ­a encontrar la mejor forma de parametrizar una red neuronal.

Se han desarrollado varios mÃ©todos para mejorar el condicionamiento del gradiente de costos para arquitecturas de redes neuronales generales. Un enfoque es usar descenso de gradiente de primer orden estÃ¡ndar sin precondicionamiento, pero cambiar la parametrizaciÃ³n del modelo para dar gradientes que son mÃ¡s parecidos a las gradientes naturales de estos mÃ©todos.

Otro enfoque en esta direcciÃ³n es la normalizaciÃ³n batch, un mÃ©todo en el que la salida de cada neurona (antes de la aplicaciÃ³n de la no linealidad) se normaliza por la media y desviaciÃ³n estÃ¡ndar de las salidas calculadas sobre los ejemplos en el minibatch. Esto reduce el desplazamiento covariable de las salidas de las neuronas.

Siguiendo este segundo enfoque para aproximar la optimizaciÃ³n del gradiente natural, se presenta un mÃ©todo simple pero general, llamado normalizaciÃ³n del peso, para mejorar la optimizaciÃ³n de los pesos de los modelos de red neuronal. El mÃ©todo se inspira en la normalizaciÃ³n batch, pero es un mÃ©todo determinista que no comparte la propiedad de la normalizaciÃ³n batch de aÃ±adir ruido a los gradientes. AdemÃ¡s, la sobrecarga impuesta por el modelo es menor: no se requiere memoria adicional y el cÃ¡lculo adicional es insignificante.


## NormalizaciÃ³n del Peso

Consideramos redes neuronales artificiales estÃ¡ndar donde el cÃ¡lculo de cada neurona consiste en tomar una suma ponderada de caracterÃ­sticas de entrada:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28w.x%20&plus;%20b%29">
</p>

donde ![](https://latex.codecogs.com/gif.latex?w) es un vector de peso k-dimensional, ![](https://latex.codecogs.com/gif.latex?b) es un tÃ©rmino de polarizaciÃ³n escalar, ![](https://latex.codecogs.com/gif.latex?x) es un vector k-dimensional de caracterÃ­sticas de entrada, ![](https://latex.codecogs.com/gif.latex?%5Cphi%28.%29) denota una funcion no lineal elemental que denota la salida escalar de la neurona. DespuÃ©s de asociar una funciÃ³n de pÃ©rdida a una o mÃ¡s salidas neuronales, dicha red neuronal es comÃºnmente entrenada por el descenso de gradiente estocÃ¡stica con los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?w), ![](https://latex.codecogs.com/gif.latex?b) y ademÃ¡s ![](https://latex.codecogs.com/gif.latex?y) denota la salida escalar de la neurona.

Con la intenciÃ³n de acelerar la convergencia de este procedimiento de optimizaciÃ³n, se realizarÃ¡ la reparameterizaciÃ³n de cada vector de peso ![](https://latex.codecogs.com/gif.latex?w) en tÃ©rminos de un vector de parÃ¡metro ![](https://latex.codecogs.com/gif.latex?v) y un parÃ¡metro escalar ![](https://latex.codecogs.com/gif.latex?g) y se calcularÃ¡ el descenso de gradiente estocÃ¡stico con respecto a esos parÃ¡metros. La expresiÃ³n del vector quedarÃ­a expresado de la siguiente forma:

<p align="center">
  <img src="https://i.imgur.com/515KiQE.png">
</p>

donde ![](https://latex.codecogs.com/gif.latex?v)  es un vector k-dimensional, ![](https://latex.codecogs.com/gif.latex?g) es un escalar, y ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%7C%7C) denota la norma euclidiana de ![](https://latex.codecogs.com/gif.latex?v). Esta reparameterizaciÃ³n tiene el efecto de fijar la norma euclidiana del vector de peso ![](https://latex.codecogs.com/gif.latex?w), siendo ahora ![](https://latex.codecogs.com/gif.latex?%7C%7Cw%7C%7C%20%3D%20g), independiente de los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?v).

## Estado del Arte

Investigaciones anteriores tambiÃ©n desarrollaban la idea de normalizar el vector de peso, pero la optimizaciÃ³n solo se realizaba mediante la parametrizaciÃ³n de ![](https://latex.codecogs.com/gif.latex?w), aplicando solamente la normalizaciÃ³n despuÃ©s de cada paso de descenso de gradiente estocÃ¡stico. Con el presente mÃ©todo se reparameteriza explÃ­citamente el modelo y realizar un descenso de gradiente estocÃ¡stico en los nuevos parÃ¡metros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g) directamente. De esta forma se mejora el condicionamiento del gradiente y conduce a una convergencia mejorada del procedimiento de optimizaciÃ³n. Al desacoplar la norma del vector de peso ![](https://latex.codecogs.com/gif.latex?g) de la direcciÃ³n del vector de peso ![](https://latex.codecogs.com/gif.latex?%28v/%7C%7Cv%7C%7C%29), se acelera la convergencia de la optimizaciÃ³n de descenso de gradiente estocÃ¡stico.

## Gradientes

El entrenamiento de una red neuronal mediante la nueva parametrizaciÃ³n se realiza utilizando mÃ©todos estÃ¡ndar de descenso gradiente estocÃ¡stico. De esta forma se obtiene el gradiente de una funciÃ³n de pÃ©rdida ![](https://latex.codecogs.com/gif.latex?L) con respecto a los nuevos parÃ¡metros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g). De la siguiente forma:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_gL%20%3D%20%5Cfrac%7B%5Cnabla_wL.v%7D%7B%7C%7Cv%7C%7C%7D">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_vL%20%3D%20%5Cfrac%7Bg%7D%7B%7C%7Cv%7C%7C%7D%5Cnabla_wL%20-%20%5Cfrac%7Bg%5Cnabla_gL%7D%7B%7C%7Cv%7C%7C%5E2%7Dv">
</p>

donde ![](https://latex.codecogs.com/gif.latex?%5Cnabla_wL) es el gradiente con respecto a los pesos ![](https://latex.codecogs.com/gif.latex?w) que se usan normalmente.

Por lo tanto, la retropropagaciÃ³n mediante la normalizaciÃ³n del peso sÃ³lo requiere una modificaciÃ³n menor de las ecuaciones habituales y se implementa fÃ¡cilmente utilizando software de red neural estÃ¡ndar, ya sea especificando directamente la red en tÃ©rminos de los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?v), ![](https://latex.codecogs.com/gif.latex?g) y dependiendo de la auto-diferenciaciÃ³n o aplicando la ecuaciÃ³n anterior en una etapa posterior al procesamiento. A diferencia de la normalizaciÃ³n batch, las expresiones anteriores son independientes del tamaÃ±o del minilote y, por tanto, causan sÃ³lo una sobrecarga computacional mÃ­nima.

Una forma alternativa de escribir el gradiente es:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_vL%20%3D%20%5Cfrac%7Bg%7D%7B%7C%7Cv%7C%7C%7DM_w%5Cnabla_wL">
</p>

con:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?M_w%20%3D%20I%20-%20%5Cfrac%7Bww%27%7D%7B%7C%7Cw%7C%7C%5E2%7D">
</p>

donde ![](https://latex.codecogs.com/gif.latex?M_w) es una matriz de proyecciÃ³n que se proyecta sobre el complemento del vector ![](https://latex.codecogs.com/gif.latex?w). Esto demuestra que la normalizaciÃ³n del peso cumple dos cosas: escala el gradiente de peso por ![](https://latex.codecogs.com/gif.latex?g/%7C%7Cv%7C%7C) y proyecta el gradiente lejos del vector de peso actual. Ambos efectos ayudan a aproximar la matriz de covarianza del gradiente a la optimizaciÃ³n de la identidad y los beneficios.

Debido a la proyecciÃ³n lejos de ![](https://latex.codecogs.com/gif.latex?w), la norma de ![](https://latex.codecogs.com/gif.latex?v) crece monotÃ³nicamente con las actualizaciones de los peso cuando una red neuronal aprende con la normalizaciÃ³n de peso usando el descenso de gradiente estÃ¡ndar sin momento. Sea ![](https://latex.codecogs.com/gif.latex?v%27%20%3D%20v%20&plus;%20%5CDelta%20v) denotando la actualizaciÃ³n de parÃ¡metros, con ![](https://latex.codecogs.com/gif.latex?%5CDelta%20v%20%5Cpropto%20%5Cnabla_vL) (ascenso / descenso mÃ¡s pronunciado), entonces ![](https://latex.codecogs.com/gif.latex?%5CDelta%20v) es necesariamente ortogonal al vector de peso actual ![](https://latex.codecogs.com/gif.latex?w), ya que nos proyectamos lejos de Ã©l al calcular ![](https://latex.codecogs.com/gif.latex?%5Cnabla_vL). Puesto que ![](https://latex.codecogs.com/gif.latex?v) es proporcional a ![](https://latex.codecogs.com/gif.latex?w), la actualizaciÃ³n es tambiÃ©n ortogonal a ![](https://latex.codecogs.com/gif.latex?v) y aumenta su norma mediante el teorema de PitÃ¡goras. EspecÃ­ficamente, si ![](https://latex.codecogs.com/gif.latex?%7C%7C%5CDelta%20v%7C%7C/%7C%7Cv%7C%7C%20%3D%20c) el nuevo vector de peso tendrÃ¡ la norma ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%27%7C%7C%20%3D%20%5Csqrt%7B%7C%7Cv%7C%7C%5E2%20&plus;%20c%5E2%7C%7Cv%7C%7C%5E2%7D%20%3D%20%5Csqrt%7B1%20&plus;%20c%5E2%7D%7C%7Cv%7C%7C%20%u2265%20%7C%7C%20v%20%7C%7C). La tasa de incremento dependerÃ¡ de la varianza del gradiente de peso. Si las gradientes son ruidosas, ![](https://latex.codecogs.com/gif.latex?c) serÃ¡ alto y la norma de ![](https://latex.codecogs.com/gif.latex?v) aumentarÃ¡ rÃ¡pidamente, lo que a su vez reducirÃ¡ el factor de escala ![](https://latex.codecogs.com/gif.latex?g/%7C%7Cv%7C%7C). Si la norma de los gradientes es pequeÃ±a, obtendremos ![](https://latex.codecogs.com/gif.latex?%5Csqrt%7B1%20&plus;%20c%5E2%7D%20%5Cthickapprox%201), y la norma de ![](https://latex.codecogs.com/gif.latex?v) dejarÃ¡ de aumentar. Usando este mecanismo, el gradiente escalado se autoestabiliza su norma.

EmpÃ­ricamente, la capacidad de crecer de la norma ![](https://latex.codecogs.com/gif.latex?%7C%7Cv%7C%7C) hace que la optimizaciÃ³n de redes neuronales con normalizaciÃ³n de peso sea muy robusta al valor de la tasa de aprendizaje. Si la tasa de aprendizaje es demasiado grande, la norma de los pesos no normalizados crece rÃ¡pidamente hasta alcanzar una tasa de aprendizaje efectiva adecuada. Una vez que la norma de los pesos ha crecido grande con respecto a la norma de las actualizaciones, la tasa de aprendizaje eficaz se estabiliza. Por lo tanto, las redes neuronales con normalizaciÃ³n de peso funcionan bien con un rango mucho mÃ¡s amplio de tasas de aprendizaje que cuando se usa la parametrizaciÃ³n normal. A su vez, las redes neuronales con normalizaciÃ³n batch tambiÃ©n poseen esta propiedad, y pueden ser explicado por este anÃ¡lisis.

## RelaciÃ³n con la normalizaciÃ³n batch

Para realizar la reparameterizaciÃ³n se usarÃ¡ la normalizaciÃ³n batch, el cual normaliza las estadÃ­sticas de la preactivaciÃ³n ![](https://latex.codecogs.com/gif.latex?t) para cada minibatch se tiene:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%27%20%3D%20%5Cfrac%7Bt%20-%20%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D">
</p>

con ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D), ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D) la media y desviaciÃ³n estÃ¡ndar de las pre-activaciones ![](https://latex.codecogs.com/gif.latex?t%20%3D%20v%20%5Ccdot%20x). Para el caso especial en el que la red sÃ³lo tiene una sola capa, y las caracterÃ­sticas de entrada ![](https://latex.codecogs.com/gif.latex?x) para esa capa son normalizadas(distribuidas independientemente con media cero y varianza unitaria), estas estadÃ­sticas son dadas por ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D%20%3D%200) y ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D%20%3D%20%7C%7Cv%7C%7C). En ese caso, la normalizaciÃ³n de las pre-activaciones mediante normalizaciÃ³n batch es equivalente a la normalizaciÃ³n de los pesos mediante la normalizaciÃ³n del peso.

Las redes neuronales convolucionales suelen tener mucho menos peso que las pre-activaciones, por lo que normalizar los pesos es a menudo mucho mÃ¡s barato computacionalmente. AdemÃ¡s, la norma de v es no estocÃ¡stica, mientras que la media de minibatch ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) y la varianza ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2%5Bt%5D) pueden tener en general una varianza alta para el tamaÃ±o de minibatch pequeÃ±o. Por lo tanto, la normalizaciÃ³n del peso puede verse como una aproximaciÃ³n mÃ¡s barata y menos ruidosa a la normalizaciÃ³n batch. Aunque la equivalencia exacta no suele mantenerse para arquitecturas mÃ¡s profundas, todavÃ­a se encuentra que el mÃ©todo de normalizaciÃ³n de peso proporciona gran parte de la aceleraciÃ³n de la normalizaciÃ³n batch completa.

## InicializaciÃ³n

AdemÃ¡s de un efecto de reparameterizaciÃ³n, la normalizaciÃ³n batch tambiÃ©n tiene el beneficio de fijar la escala de las caracterÃ­sticas generadas por cada capa de la red neuronal. Esto hace que la optimizaciÃ³n sea robusta frente a las inicializaciones de parÃ¡metros para las cuales estas escalas varÃ­an entre capas. Dado que la normalizaciÃ³n del peso carece de esta propiedad, es importante inicializar adecuadamente estos parÃ¡metros. Entonces se realizarÃ¡ un muestreo de los elementos de ![](https://latex.codecogs.com/gif.latex?v) mediante una distribuciÃ³n simple con una escala fija, por ejemplo: distribuciÃ³n normal con media cero y desviaciÃ³n estÃ¡ndar 0,05. Antes de iniciar el entrenamiento, se inicializa los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?b) y ![](https://latex.codecogs.com/gif.latex?g) para fijar las estadÃ­sticas de minibatch de todas las pre-activaciones en la red, como en la normalizaciÃ³n batch, pero solo para un Ãºnico minibatch de datos y sÃ³lo durante la inicializaciÃ³n. Esto se puede realizar de manera eficiente realizando un primer paso de feedforward a travÃ©s de nuestra red para un Ãºnico minibatch de datos ![](https://latex.codecogs.com/gif.latex?X), usando el siguiente cÃ¡lculo en cada neurona:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%20%3D%20%5Cfrac%7Bv%5Ccdot%20x%7D%7B%7C%7Cv%7C%7C%7D">
</p>

y

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28%5Cfrac%7Bt%20-%20%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D%29">
</p>

donde ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) y ![](https://latex.codecogs.com/gif.latex?%5Csigma%5Bt%5D) son la media y la desviaciÃ³n estÃ¡ndar de la preactivaciÃ³n ![](https://latex.codecogs.com/gif.latex?t) sobre los ejemplos en el minibatch. Entonces podemos inicializar el sesgo ![](https://latex.codecogs.com/gif.latex?b) de la neurona y la escala ![](https://latex.codecogs.com/gif.latex?g) como:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?g%20%5Cleftarrow%20%5Cfrac%7B1%7D%7B%5Csigma%5Bt%5D%7D%2C%20b%20%5Cleftarrow%20%5Cfrac%7B-%5Cmu%5Bt%5D%7D%7B%5Csigma%5Bt%5D%7D">
</p>

de manera que ![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28w.x%20&plus;%20b%29). Como la normalizaciÃ³n batch, este mÃ©todo asegura que todas las caracterÃ­sticas inicialmente tienen media cero y varianza unitaria antes de la aplicaciÃ³n de la no linealidad. Con el presente mÃ©todo esto sÃ³lo es vÃ¡lido para el minibatch que utilizamos para la inicializaciÃ³n, y los minibatches subsiguientes pueden tener estadÃ­sticas ligeramente diferentes, pero experimentalmente se encuentra que este mÃ©todo de inicializaciÃ³n funciona bien. El mÃ©todo tambiÃ©n puede aplicarse a redes sin normalizaciÃ³n de peso, simplemente haciendo una optimizaciÃ³n de gradiente estocÃ¡stico en los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?w) directamente, despuÃ©s de la inicializaciÃ³n en tÃ©rminos de ![](https://latex.codecogs.com/gif.latex?v) y ![](https://latex.codecogs.com/gif.latex?g). La desventaja de este mÃ©todo de inicializaciÃ³n es que sÃ³lo puede aplicarse en casos similares donde la normalizaciÃ³n de lote es aplicable. Para modelos con recursividad, como RNNs y LSTMs, tendremos que recurrir a mÃ©todos de inicializaciÃ³n estÃ¡ndar.

<p align="center">
  <img src="https://i.imgur.com/JCuSUuv.png">
</p>


## NormalizaciÃ³n Batch

La normalizaciÃ³n del peso, hace que la escala de las activaciones neuronales sea aproximadamente independiente de los parÃ¡metros ![](https://latex.codecogs.com/gif.latex?v). Sin embargo, a diferencia de la normalizaciÃ³n batch, las medias de las activaciones neuronales siguen dependiendo de ![](https://latex.codecogs.com/gif.latex?v). con una versiÃ³n especial de la normalizaciÃ³n batch, a la que llamamos normalizaciÃ³n batch only mean. Con este mÃ©todo de normalizaciÃ³n, se resta las medias del minibatch como con la normalizaciÃ³n batch completa, pero divide por las desviaciones estÃ¡ndar de minibatch. Es decir, calculamos las activaciones neuronales utilizando:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%20%3D%20w%5Ccdot%20x">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?t%27%20%3D%20t%20-%20%5Cmu%5Bt%5D%20&plus;%20b">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cphi%28t%27%29">
</p>


donde ![](https://latex.codecogs.com/gif.latex?w) es el vector de peso, parametrizado usando la normalizaciÃ³n del peso, y ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) es la media del minibatch de la preactivaciÃ³n ![](https://latex.codecogs.com/gif.latex?t). Durante el entrenamiento, mantenemos un promedio de la media del minibatch que sustituimos por ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%5D) en el tiempo de prueba.

<p align="center">
  <img src="https://i.imgur.com/aQ8w9UG.png">
</p>

El gradiente de la pÃ©rdida con respecto a la preactivaciÃ³n ![](https://latex.codecogs.com/gif.latex?t) se calcula como:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_tL%20%3D%20%5Cnabla_%7Bt%27%7DL%20-%20%5Cmu%5B%5Cnabla_%7Bt%27%7DL%5D">
</p>


donde ![](https://latex.codecogs.com/gif.latex?%5Cmu%5B%5Ccdot%20%5D) denota una vez mÃ¡s la operaciÃ³n para tomar la media del minibatch. Por lo tanto, la normalizaciÃ³n batch de media sÃ³lo tiene el efecto de centrar los gradientes que son retropropagados. Esta es una operaciÃ³n comparativamente barata, y la sobrecarga computacional de la normalizaciÃ³n batch de sÃ³lo media es por lo tanto menor que para la normalizaciÃ³n batch completa. AdemÃ¡s, este mÃ©todo produce menos ruido durante el entrenamiento, y el ruido que se produce es mÃ¡s suave, ya que la ley de grandes nÃºmeros asegura que ![](https://latex.codecogs.com/gif.latex?%5Cmu%5Bt%20%5D) y ![](https://latex.codecogs.com/gif.latex?%5Cmu%5B%5Cnabla%20t%5D) estÃ¡n aproximadamente distribuidos normalmente.



