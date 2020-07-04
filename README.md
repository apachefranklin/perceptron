# perceptron
that repository is the implementation from scratch of perceptron

# general_utility_ml.py
the general.py file contains the functions to share between all 
the machine learning algorithms, much more with regard to the evaluations.
Actualy, it have only confusion_matrix function who take the value predicted
and real and return dictionnary **{"confusion":confusion_matrix,"namedconfusion":named_confusion_matrix}**
The difference is that **confusion** is a numpy array wihout name on column and **named_confusion** is a **pandas dataframe**
who have columns names and row names for excelent visualition in console of our confusion matrix

# perceptron_monocouche.py

that file contain a main perceptron class
when you wan to initialize you have parameters like
  ## x
    how data future
  ## y
    how target
  ## theta
    the bias
  ## activate_function
    you can choose sigmoid,relu,urelu,tangent or heavisde
    by default sigmoid is use like activate function
  ## learning_rate
    for the update of our weight vector
  ## accept_error
    error when the model is below, it considering convergence
  ## epoch
    The number of times, when model loop of our data, it stop running and consider all current parameter
Like function we have different function on our model, **init_weigth** initialize random weight for our trainning and modify directly the
**self.weight** for perceptron objects. The function **__update_weight** who update wieght depends of wich line we are in our data, and the current_error

the function train for training, the function **predict_single_future** who take a set of future and the **line** you want to predicc, make prediction
the function **test** who return an dictionnary **{"prediction":predict_target,"confusion":confusion_matrix}**
The function **predict** who can be use in production envrironnment
