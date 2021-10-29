# Clases del Curso de Fundamentos de Redes Neuronales con Python y Keras

## La importancia de las redes neuronales en la actualidad

Keras no es un backend es un API que se conecta con un backend como
Tensorflow, Theano, CNTK, PyTorch

En este curso usaremos Keras con Tensorflow

### Resumen de la clase Estudiantes

- Las herramientas más conocidas para manejar redes neuronalnes son TensorFlow y PyTorch.
- Keras es una API, se utiliza para facilitar el consumo del backend.
- Utilizaremos la tarjeta GPU, porque permite procesas más datos matemáticos necesarios en el deep learning.

## ¿Qué es deep learning?

La inteligencia artificial son los intentos de replicar la inteligencia humana en sistemas artificiales.

**Machine learning** son las técnicas de aprendizaje automático, en donde mismo sistema aprende como encontrar una respuesta sin que alguien lo este programando.

**Deep learning** es todo lo relacionado a las redes neuronales. Se llama aprendizaje profundo porque a mayor capas conectadas ente sí se obtiene un aprendizaje más fino.

En el Deep learning existen dos grandes problemas:

1. **Overfitting**: Es cuando el algoritmo “memoriza” los datos y la red neuronal no sabe generalizar.

2. **Caja negra**: Nosotros conocemos las entradas a las redes neuronales. Sim embargo, no conocemos que es lo que pasa dentro de las capas intermedias de la red.

En un algoritmo de Machine Learning es mucho mas facil explicar como funciona internamente, mientras que las redes neuronales suelen ser cajas negras a no se de trabajar en lograr comprenderlas lo cual requiere un esfuerzo considerable

Es incluso una discusion de etica si es responsable usar las redes neuronales cuando no sabemos como funcionan internamente.

[Visualizacion 3D de una red neuronal](https://www.cs.ryerson.ca/~aharley/vis/conv/)

## La neurona: una pequeña y poderosa herramienta

La neurona (perceptron)
fue inspirado por las redes neuronales biologicas

El perceptron tiene
entradas de datos x1 ... xn
tenemos pesos para cada entrada de datos
dentro de la neurona tenemos sumas ponderadas

Tenemos la entrada de datos y los pesos de las neuronas, dentro de la neurona se realiza una suma ponderada y a ese resultado se le realiza una funcion de activacion la cual nos da un resultado

En cada iteracion de la red neuronal se cambian los pesos de la entrada para ir optimizando la funcion

Si el nivel de complejidad de un problema es mucho para que la conversion lineal de una neurona lo solucione ejemplo una neuroana XOR el uso de una capa extra de neuronas permite que se pueda resolver el problema.

## Arquitectura de una red neuronal

Una red neuronal esta compuesta de capaz
cada capa agrega informacion a la siguiente capa de la red

La capa inicial de entrada es el **Input Layer**, todas la capaz entrea la inicial y la final son las capas ocultas o **Hidden Layers** y la capa final es la capa de salida o **Output Layer**

Cada capa de las red neuronal trabaja con caracteristicas que le fueron pasada de la red anterior

la operacion de una capa de una red a la siguiente se puede comprender como producto la multiplicacion entre una matriz y un vector

y luego se le suma el bias el cual es una constante cuyo valor cambia igual que los pesos

dentro de una red neuronal ocurren miles de operciones de producto punto de matrices, y en cada iteracion se recalculan el valor de los pesos y constantes que se utilizaran en cada operacion.

Existe un problema matematico que si todas las operaciones son lineales no importa la cantidad de capaz el resultado seria una linea la cual perderia todo el aprendizaje hecho, para evitar esto las redes neuronales cuentan con las "funciones de activacion"

[Neural Network playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.98911&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## Funciones de activación

Existen porque no se pueden aplicar sumatorias consecutivas de lineas ya que el resultado seria una linea, perdiendo el aprendizaje hecho.

Tipos de funciones de activación:

- Discretas
  Conjunto finito de valores

- Continuas
  Valores dentro de algun intervalo

### Funciones Discretas

- Función Escalonada
  si el valor es mayor a 0 da 1 si es menor que 0 da 0
  $$
  \begin{cases}
    1 \text{ si } z >= 0 \\
    0 \text{ si } z < 0
  \end{cases}
  $$

- Función signo/signum
  $$
  \begin{cases}
    1 \text{ si } z >= 0 \\
    -1 \text{ si } z < 0
  \end{cases}
  $$

### Funciones Continuas

- Función Sigmoidal/sigmoid
  da valores continuos entre 0 y 1
  muy buena para el calculo de probabilidades
  ademas de podersele aplicar derivadas
  $$
  S(x)= \frac {1}{1+e^{-x}}
  $$

- Función tangente hiperbólica/tanh
  Da valores continuos entre -1 y 1.

  $$
   tanh ( x ) = sinh ( x ) cosh ( x ) = e 2 x − 1 e 2 x + 1 .
  $$

- Función rectificada/ReLU (Rectified Linear Units)
  es de las funciones mas usadas actualmente
  puede derivarse
  $$
  \begin{cases}
    z \text{ si } z >= 0 \\
    0 \text{ si } z < 0
  \end{cases}
  $$

- Función Softmax
  Da la probabilidad de cada una de las posibles salidas
  se utiliza mucho para realizar calificacion

  $$
  \sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
  $$

[Herramienta para la ciencia de datos y funciones de activación](https://www.wolframalpha.com/)

## Función de pérdida (loss function)

La función de pérdida es muy importante definirla

Se encarga de decirnos si nuestro algoritmo esta prediciendo de una manera correcta, la función toma la prediccion y los valores reales y nos da un score en base a los resultados y asi saber que tan grande es el error de la prediccion

- MSE - Mean Squared error (Error cuadratico Medio)
  Eleva los valores al cuadrado para castigar por los valores que estan mas alejados
  $$
  \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n}(Y_{i}-\hat{Y}_{i})^2
  $$

- Cross Entropy
  Es muy bueno para valores categoricos o de clasificacion

  Se encarga de medir la distancia entre la prediccion real y el valor de nuestro algoritmo

  $$
    -\sum_{x}p(x) \log{q(x)}
  $$

## Descenso del gradiente

Con la derivada cuando la pendiente da 0 es porque se llego a un pico o a un valle, por lo cual de forma iterativa se busca optimizar la función para encontrar el menor valor de la funcion

http://www.benfrederickson.com/numerical-optimization/

https://www.desmos.com/calculator/l0puzw0zvm

El learning Rate de una funcion dice que tantos pasos realiza la funcion para buscar su punto optimo, si el learning rate es muy bajo la funcion deja de ser eficiente y si el learning rate es muy alto la funcion falla en encontrar el valor minimo al pasar de largo por este punto y no tener la presicion suficiente como para alcanzarlo

### Mínimos

Si la funcion alcanza un minimo local pero este no es el minimo global, la funcion esta optmizada mas no optimizada al maximo

para esto casos existen opmizadores como el **MSPro** el cual le agrega momento y comportamiento fisico a la funciones para evitar que se estanquen en un minimo local

el descenso del gradiente se calcula tomando la funcion de perdida y se calculan sus derivadas parciales en un punto, luego este vector gradiente resultante nos indica hacia donde hay que ir para aumentar el valor al maximo por lo cual lo multiplicamos por -1 y al avanzar en la direccion opuesta realizamos el descenso mas rapido

[DotCSV Descenso del gradiente](https://www.youtube.com/watch?v=A6FiCDoz8_4)

## Backpropagation

Backpropagation, [teoría](https://www.youtube.com/watch?v=eNIqz_noix8)
Backpropagation, [matemáticas](https://www.youtube.com/watch?v=M5QHwkkHgAA)

Es el algoritmo que se encarga de distribuir el error calculado por la funcion de partidad a traves de toda la red neuronal, recorriendola red iniciando desde la capa final hasta la capa inicial de entrada.

[backpropagation brilliant](https://brilliant.org/wiki/backpropagation/#)

### Math

For the partial derivatives,

$$\frac{\partial E_d}{\partial w_{ij}^k} = \delta_j^k o_i^{k-1}.$$

For the final layer's error term,

$$\delta_1^m = g_o^{\prime}(a_1^m)\left(\hat{y_d}-y_d\right).$$

For the hidden layers' error terms,

$$\delta_j^k = g^{\prime}\big(a_j^k\big)\sum_{l=1}^{r^{k+1}}w_{jl}^{k+1}\delta_l^{k+1}.$$

For combining the partial derivatives for each input-output pair,

$$\frac{\partial E(X, \theta)}{\partial w_{ij}^k} = \frac{1}{N}\sum_{d=1}^N\frac{\partial}{\partial w_{ij}^k}\left(\frac{1}{2}\left(\hat{y_d} - y_d\right)^{2}\right) = \frac{1}{N}\sum_{d=1}^N\frac{\partial E_d}{\partial w_{ij}^k}.$$

For updating the weights,

$$\Delta w_{ij}^k = - \alpha \frac{\partial E(X, \theta)}{\partial w_{ij}^k}.$$

## Playground - Tensorflow

[Playground de Tensorflow](https://playground.tensorflow.org/)

[Espiral con relu](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=6,6,6,4&seed=0.76006&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false)

[Espiral con tangente hiperbolica](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=6,4,4&seed=0.82058&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

[Dona sin neuronas](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=&seed=0.20410&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## Dimensiones, tensores y reshape

Tipos de datos por dimension

- Scalar: dimension 0, ejemplo: `2`
- Vector: dimension 1, ejemplo `[1,2,3]`
- Matrix: dimension 2 , ejemplo `[[1,2],[3,4]]`
- Tensores: 3 Dimensiones o N dimensiones, ejemplo
  `[[[1,2],[4,5]],[[1,2],[4,5]]]`

### Matrix 2D

Matriz, dataframe de filas y columnas

### Tensor 3D

Podemos usar series de Tiempo

Donde tenemos

1. cantidad de ejemplos en 1 dimension
2. caracteristica de esos ejemplos en otra dimension
3. caractersisticas de esos ejemplos con respecto al tiempo en una tercera dimension

### Tensor 4D Images

1. Ancho
2. Alto
3. Canales de Colores (RGB)
4. Cantidad de Ejemplos

## Entrenamiento forward de la red neuronal

https://machinelearningknowledge.ai/

https://machinelearningknowledge.ai/wp-content/uploads/2019/06/Artificial-Neuron-Working.gif

## Aplicando backpropagation y descenso del gradiente

## Data: train, validation, test

### Manejo de datos

Aunque comumente se dividen los datos en 2

- Entrenamiento
- Test

Para el deep learning y buena practica de machine learning es mejor divirlos en 3

- Entrenamiento

- Validacion
  Con el test de validacion se nos da la oportunidad de cambiar hiperparametros para mejorar como puede ser el learning rate, numero de epochs, etc..

- Test
  Con el test ya se usa al final para evaluar como es el comportamiento de mi modelo
  Testear el resultado final accuracy o unidad de medida que se utilice

¿Por qué hacer esto?
por cuestiones de etica para que el modelo se ponga a prueba con datos que jamas allá visto

### Notas alumnos

Racismo por IA:
Como dijo el profesor anteriormente, han existido diversos dilemas éticos a lo largo de la historia reciente en los que se describe cómo la Inteligencia Artificial puede tener y ha tenido diversos sesgos que pueden convertir a sus resultados en lo que en un humano pasarían por ser racistas o discriminatorios de alguna forma.

El problema es que estos sesgos, al no poder evaluar detalladamente los ingenieros de software el funcionamiento interno de estos modelos de redes neuronales, no pueden corregirse y pasan desapercibidos.

Un ejemplo:

En un banco, se solicita a un ingeniero de software que haga un modelo que prediga mediante deep learning qué tan probable es recibir beneficios en el banco de otorgar un préstamo a una persona.

Este algoritmo tomaría en cuenta el nivel socioeconómico históricamente inferior de las personas afroamericanas y africanas y decide que tener esta procedencia o raza es una variable de peso que descarta (o hace menos probable) que a esa persona se le otorgue un préstamo.

Esto es un problema, porque el algoritmo no se equivocó, los datos en los que se fundamenta son históricamente reales, sin embargo, es injusto que el banco aplique sistemáticamente esta postura puesto que ello perpetuará esta condición de pobreza o nivel socioeconómico medio-bajo en estas personas (al no recibir tener préstamos fácilmente, no tendrán tantas alternativas para salvar sus negocios y empresas y tenderán más a la bancarrota, etc.).

Es por eso que se ha pensado en prohibir el uso de estos modelos de funcionamiento de caja negra en la toma de cierto tipo de decisiones que podrían resultar en este tipo de conflictos, además que se continúan buscando mecanismos efectivos que puedan tomar en cuenta y corregir este tipo de dilemas éticos.

## Resolviendo un problema de clasificacion binaria
