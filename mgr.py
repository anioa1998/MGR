import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
 

def prepareResultDataframe(x): 
    resultDateFrame = pd.DataFrame(index = x.index.copy()) 
    return resultDateFrame 

def prepareData(dataframe): 
    x = pd.DataFrame(dataframe.iloc[:, 0:4]) 
    y = pd.DataFrame(dataframe.iloc[:,4]) 

    y = LabelEncoder().fit_transform(y) 
    x = MinMaxScaler().fit_transform(x) 

    x = pd.DataFrame.from_records(x) 

    return x,y 


def oppositeClassCount(x, y, k, resultDateFrame): 
    neigh = NearestNeighbors(n_neighbors=k+1) 
    neigh.fit(x,y)

    newColumnName = f"knn{k}"
    resultDateFrame[newColumnName] = np.nan

    for index, row in x.iterrows():

        #Set knn  
        neighbors_ids = neigh.kneighbors([row], return_distance=False)
        neighbors_ids = list(neighbors_ids[0])
        neighbors_ids.remove(index)

        row_type = y[index] 
        counter = 0 
    
        #Get type of neighbors 
        for i in neighbors_ids: 
        	if y[i] != row_type: 
                     counter = counter + 1 
        resultDateFrame.at[index, newColumnName] = counter

        #Aktualnie (knn3) 5x 3 sąsiadów przeciwnej klasy - bazujemy na poprawnych danych y, jak wyodrębnić x i y bez testowanego obiektu?

def meanDistanceFromNN(x, y, k, resultDateFrame):
    neigh = NearestNeighbors(n_neighbors=k+1) 
    neigh.fit(x,y)

    newColumnName = f"meanDistance{k}"
    resultDateFrame[newColumnName] = np.nan

    for index, row in x.iterrows():

        #Set knn  
        neighbors_distance = neigh.kneighbors([row], return_distance=True)

        neighbors_distance = list(neighbors_distance[0][0])
        neighbors_distance.remove(0.0)
    
        #Calculate mean distance
        mean = np.mean(neighbors_distance)
        resultDateFrame.at[index, newColumnName] = mean

def smallestDistanceSameClass(x, y, k, resultDateFrame):
     #xxxx
    test = 'x'

def smallestDistanceAnyClass(x, y, k, resultDateFrame):
    #xxxx
    test = 'x'

irisset = pd.read_csv( 
    filepath_or_buffer="Iris.csv", 
    sep=",", 
    encoding="utf-8") 

irisset = irisset.set_index("Id") 
x, y = prepareData(irisset) 
resultDataFrame = prepareResultDataframe(x)
k_values = list([3,5,7,8,9,11,12,14,15,17])
for k in k_values:
    oppositeClassCount(x, y, k, resultDataFrame)
for k in k_values:
    meanDistanceFromNN(x, y, k, resultDataFrame)
s = "test"