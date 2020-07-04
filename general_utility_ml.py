import numpy as np
import pandas as pd
class GeneralUtiliy:
    def confusion_matrix(self,data_x,data_y):
        data_x=np.array(data_x)
        data_y=np.array(data_y)
        unique_y=np.unique(data_y)
        unique_x=np.unique(data_x)
        j=0
        pos_dict={}
        for nb in unique_x:
            pos_dict[str(nb)]=j
            j+=1
        """
            that function take two array and compute
            confusion matrix of them
            on doit avoir une matrice carree
        """
        lenght=len(unique_x)
        confusition_matrix=np.zeros((lenght,lenght),dtype=int)
        i=0
        for nb_x in data_x:
            nb_y=data_y[i]
            pos_x=pos_dict[str(nb_x)] #position de x
            pos_y=pos_dict[str(nb_y)] #pos_y
            confusition_matrix[pos_x,pos_y]+=1
            i=i+1
        list_with_tuple=[]
        for name in pos_dict.keys():
            print(confusition_matrix[pos_dict[name]])
            element=(name,confusition_matrix[pos_dict[name]])
            list_with_tuple.append(element)
        confusion_matrix_with_name=pd.DataFrame(data=confusition_matrix,columns=unique_x)

        return {"confusion":confusition_matrix,"namedconfusion":confusion_matrix_with_name}