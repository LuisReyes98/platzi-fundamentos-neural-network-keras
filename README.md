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

- Función rectificada/ReLU
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
