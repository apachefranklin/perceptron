import math
import numpy as np
import random
from general_utility_ml import *
"""Cette classe doit avoir une fonction qui fait la somme des elements d'une ligne
   Une fonction qui calcule la somme des wi xi
   Une fonction qui met a jour les poids
   @author apachefranklin
   """
class Perceptron:
    """@author apachefranklin"""
    def __init__(self,x,y,theta=0.1,activate_function="sigmoid",learning_rate=0.25,accept_error=0.1,epoch=1):
        self.weight=[]
        self.y=np.array(y)
        self.x=np.array(x)
        self.theta=theta
        self.learning_rate=learning_rate
        self.accept_error=accept_error
        self.epoch=epoch
        self.shape=self.x.shape
        self.weight_list=[]
        self.activate_function=activate_function

    

    def sigmoid(self,x):
        """Fonction d'activation sigmoid"""
        return 1/(1+math.exp(-x))
    def heavyside(self,x):
        """Fonction d'activation heavyside"""
        return 1 if x>=0 else 0
    
    def hyperbolique(self,x):
        """Fonction d'activation tangente hyperbolique """
        expo_x=math.exp(-x)
        return (1-expo_x)/(1+expo_x)
    
    def relu(self,x):
        return 0 if x<0 else x
    
    def like_relu(self,x):
        return max(0.1*x,x)
    
    def sum_line_theta(self,line):
        """Cette fonction cacule la somme des Wi*Xi"""
        summ=0.0
        #print(self.weight)
        for i in list(range(self.shape[1])):
            #print(float(self.x[line][i]))
            #print(self.weight[i])
            summ=summ+float(self.x[line][i])*float(self.weight[i])
        print(summ)
        print(self.weight)
        return summ
    def init_weigth(self):
        """Cette fonction initialize les poids """
        size=self.x.shape[1]
        for i in list(range(size)):
            self.weight.append(random.random())
        print(self.weight)
    
    def _update_weight(self,current_error,line):
        """cette fonction fait une mise ajour des poids de 
        Notre perceptron"""
        lenght=len(self.weight)
        for i in list(range(lenght)):
            w=self.weight[i]
            self.weight[i]=w*self.learning_rate*current_error*self.x[line][i]

    def train(self):
        """Fonction who make training for how data """
        error=self.accept_error+1
        k=0
        self.init_weigth()
        while error>self.accept_error and k<=self.epoch:
            error=0
            for i in list(range(self.shape[0])):
                #print(self.sum_line_theta(i))
                sum_weigth=self.sum_line_theta(i)-self.theta
                activation=0
                function=self.activate_function
                if(function=="relu"):
                    activation=self.relu(sum_weigth)
                elif(function=="urelu"):
                    activation=self.like_relu(sum_weigth)
                elif(function=="tangent"):
                    activation=self.hyperbolique(sum_weigth)
                else:
                    activation=self.sigmoid(sum_weigth)
                current_error=self.y[i]-activation
                error=error+current_error
                #Now we can update different weight
                self._update_weight(current_error,i)
                self.weight_list.append(self.weight)
                k+=1
    
    def _predict_single_feature(self,data_x,line_i):
        single_feature=data_x[line_i]
        target=0
        i=0
        for nb in single_feature:
            target=target+nb*self.weight[i]
            i+=1
        return target
    
    def predict(self,data_x):
        """That function help to make prediction in the model 
            Predicion consist to multipy evriy column by the homologue
            in the weight vector
        """
        data_x=np.array(data_x)
        dimension=self.shape[2]
        data_dimension=data_x.shape[2]
        prediction_array=[]
        if(data_dimension==dimension):
            for i in range(data_x.shape[0]):
                prediction_array.append(self._predict_single_feature(data_x,i))
            return prediction_array
        else:
            return False
        pass
    
    def test(self,data_x,data_y):
        """That function will test data and return an array
        of y predict
           With confusionmatrix 
           For do it we will make prediction function 
           to do it
        """
        data_predict=self.predict(data_x)
        confusion_matrix=GeneralUtiliy().confusion_matrix(data_predict,data_y)
        data={"prediction":data_predict,"confusion":confusion_matrix}
        return data;
        



data_x=[[0,0],
        [0,1],
        [1,0],
        [1,1]]
#print(data_x)
data_y=[0,1,1,1]

perceptron=Perceptron(data_x,data_y,theta=1)

perceptron.train()

"""perceptron=Perceptron(45,43)
#per2=Perceptron(34,45)
#print("Heavisaide de 4 donne :"+str(per2.heavyside(4)))
print("Heaviyde de -4 donne :"+str(per2.heavyside(-4)))

print("Sigmoig de 4 donne :"+str(per2.sigmoid(4)))
print("Sigmoid de -4 donne :"+str(per2.sigmoid(-4)))

bonjour=[
    [23,34,56,34],
    [12,13,15,17],
    [1,2,7,8]
]
bonjour=np.array(bonjour)
print("Now test the selection")
print(bonjour[2])
print(bonjour.shape)
#print(list(range(1,6)))

gen=GeneralUtiliy()
data_x=[1,1,0,0,1,0]
data_y=[0,1,1,0,1,0]
test_dimension=np.zeros((5,5),dtype=int)
#print(test_dimension)
print("L'heure des vraies sorties")
print(gen.confusion_matrix(data_x,data_y)['namedconfusion'])"""