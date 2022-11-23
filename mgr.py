import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
 

def prepareData(dataframe): 
    x = pd.DataFrame(dataframe.iloc[:, 0:4]) 
    y = pd.DataFrame(dataframe.iloc[:,4]) 

    y = LabelEncoder().fit_transform(y) 
    x = MinMaxScaler().fit_transform(x) 

    x = pd.DataFrame.from_records(x) 

    return x,y 

def prepareResultDataframe(x): 
    result_dataframe = pd.DataFrame(index = x.index.copy()) 
    return result_dataframe 

def prepareKNN(result_dataframe, new_column_name, x, y, k):
    neigh = NearestNeighbors(n_neighbors=k+1) 
    neigh.fit(x,y)

    result_dataframe[new_column_name] = np.nan
    return neigh, new_column_name


def oppositeClassCount(x, y, k, result_dataframe): 

    neigh, new_column_name = prepareKNN(result_dataframe, f"oppositeClassNeighbors{k}", x, y, k)

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
        result_dataframe.at[index, new_column_name] = int(counter)

        #Aktualnie (knn3) 5x 3 sąsiadów przeciwnej klasy - bazujemy na poprawnych danych y, jak wyodrębnić x i y bez testowanego obiektu?

def meanDistanceFromNN(x, y, k, result_dataframe):

    neigh, new_column_name = prepareKNN(result_dataframe, f"meanDistance{k}", x, y, k)

    for index, row in x.iterrows():

        #Set knn  
        neighbors_distance = neigh.kneighbors([row], return_distance=True)
        neighbors_distance = list(neighbors_distance[0][0])
        neighbors_distance.remove(0.0)
    
        #Calculate mean distance
        mean = np.mean(neighbors_distance)
        result_dataframe.at[index, new_column_name] = mean

def smallestDistanceSameClass(x, y, k, result_dataframe):

    neigh, new_column_name = prepareKNN(result_dataframe, f"smallestDistanceSameClass", x, y, k)
    for index, row in x.iterrows():

        #Set knn  
        neighbors = neigh.kneighbors([row], return_distance=True)
        neighbors_ids = list(neighbors[1][0])
        neighbors_distance = list(neighbors[0][0])
        neighbors_ids.remove(index)
        neighbors_distance.remove(0.0)

        smallest_distance = np.nan

        neighbors_dictionary = dict(zip(neighbors_ids, neighbors_distance))
        row_type = y[index]
        for i in neighbors_ids: 
        	if y[i] == row_type: 
                    smallest_distance = neighbors_dictionary[i]
        result_dataframe.at[index, new_column_name] = smallest_distance


def smallestDistanceAnyClass(x, y, k, result_dataframe):

    neigh, new_column_name = prepareKNN(result_dataframe, f"smallestDistanceAnyClass", x, y, k)
    for index, row in x.iterrows():
        #Set knn  
        neighbors = neigh.kneighbors([row], return_distance=True)
        neighbors_distance = list(neighbors[0][0])
        neighbors_distance.remove(0.0)
        result_dataframe.at[index, new_column_name] = neighbors_distance[0]


irisset = pd.read_csv( 
    filepath_or_buffer="Iris.csv", 
    sep=",", 
    encoding="utf-8") 

irisset = irisset.set_index("Id") 
x, y = prepareData(irisset) 
resultDataFrame = prepareResultDataframe(x)
k_values = list([3,5,7])
for k in k_values:
    oppositeClassCount(x, y, k, resultDataFrame)
    meanDistanceFromNN(x, y, k, resultDataFrame)
    smallestDistanceSameClass(x, y, k, resultDataFrame)
    smallestDistanceAnyClass(x, y, k, resultDataFrame)
s = "test"