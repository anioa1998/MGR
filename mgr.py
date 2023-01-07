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

def prepareKNN(x, y, k):
    neigh = NearestNeighbors(n_neighbors=k+1, metric="euclidean") 
    neigh.fit(x,y)
    
    return neigh

def getNearestNeighbours(neigh, index, row):
    neighbors = neigh.kneighbors([row], return_distance=True)
    neighbors_ids = list(neighbors[1][0])
    neighbors_distance = list(neighbors[0][0])
    neighbors_ids.remove(index)
    neighbors_distance.pop(0)

    return neighbors_ids, neighbors_distance

def createNewColumn(result_dataframe, new_column_name):
    result_dataframe[new_column_name] = np.nan

def setKColumns(k, result_dataframe):
    new_columns = ["oppositeClassNeighbors", "sameClassNeighbors", "meanDistanceAny", "meanDistanceSame"]
    for column in new_columns:
        column = column + str(k)
        createNewColumn(result_dataframe, column)

def setSingleColumns(result_dataframe):
    new_columns = ["minDistanceSameClass", "minDistanceAnyClass", "minDistanceOppositeClass"]
    for column in new_columns:
        createNewColumn(result_dataframe, column)

def getNeighbourTypesById(y, neighbors_ids):
    result_dict = dict()
    for id in neighbors_ids:
        result_dict[id] = y[id]

    return result_dict

def distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types: dict, row_type):
    
    neighbors_dictionary = dict(zip(neighbors_ids, neighbors_distance))
    filtered_types_dict = {k : v for k, v in neighbors_types.items() if v == row_type}
    return [neighbors_dictionary[i] for i in filtered_types_dict.keys()]

def distanceFilteredByOppositeType(neighbors_ids, neighbors_distance, neighbors_types: dict, row_type):
    
    neighbors_dictionary = dict(zip(neighbors_ids, neighbors_distance))
    filtered_types_dict = {k : v for k, v in neighbors_types.items() if v != row_type}
    return [neighbors_dictionary[i] for i in filtered_types_dict.keys()]

#----------------------------------------------------------------------------------------------------------

def oppositeClassCount(index, neighbors_types: dict, result_dataframe, row_type, k): 

    counter = 0
    #Get type of neighbors 
    for neighbor_type in neighbors_types.values(): 
        if neighbor_type != row_type: 
            counter = counter + 1 
    result_dataframe.at[index, f"oppositeClassNeighbors{k}"] = int(counter)

def sameClassCount(index, neighbors_types: dict, result_dataframe, row_type, k): 
    
    counter = 0
    #Get type of neighbors 
    for neighbor_type in neighbors_types.values(): 
        if neighbor_type == row_type: 
            counter = counter + 1 
    result_dataframe.at[index, f"sameClassNeighbors{k}"] = int(counter)

        #Aktualnie (knn3) 5x 3 sąsiadów przeciwnej klasy - bazujemy na poprawnych danych y, jak wyodrębnić x i y bez testowanego obiektu?

def meanDistanceFromAny(index, neighbors_distance, k, result_dataframe):

    #Calculate mean distance
    mean = np.mean(neighbors_distance)
    result_dataframe.at[index, f"meanDistanceAny{k}"] = mean

def meanDistanceFromSame(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, k, result_dataframe):
    mean = np.nan

    if row_type not in neighbors_types.values():
        return

    distance_list = distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types, row_type)

    mean = np.mean(distance_list)
    result_dataframe.at[index, f"meanDistanceSame{k}"] = mean
    

def smallestDistanceSameClass(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, result_dataframe):

    if str(result_dataframe.at[index, "minDistanceSameClass"]) == 'nan':
        
        if row_type not in neighbors_types.values():
            return

        distance_list = distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types, row_type)
        result_dataframe.at[index, "minDistanceSameClass"] = min(distance_list)
        
def smallestDistanceOppositeClass(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, result_dataframe):

    if str(result_dataframe.at[index, "minDistanceOppositeClass"]) == 'nan':
        
        if all(type_from_list == row_type for type_from_list in neighbors_types.values()):
            return

        distance_list = distanceFilteredByOppositeType(neighbors_ids, neighbors_distance, neighbors_types, row_type)
        result_dataframe.at[index, "minDistanceOppositeClass"] = min(distance_list)


def smallestDistanceAnyClass(neighbors_distance, index, result_dataframe):

    if str(result_dataframe.at[index, "minDistanceAnyClass"]) == 'nan':
        result_dataframe.at[index, "minDistanceAnyClass"] = min(neighbors_distance)


irisset = pd.read_csv( 
    filepath_or_buffer="Iris.csv", 
    sep=",", 
    encoding="utf-8") 

irisset = irisset.set_index("Id") 
x, y = prepareData(irisset) 
result_dataframe = prepareResultDataframe(x)
k_values = list([3,5,7])

for k in k_values:
    neigh = prepareKNN(x,y,k)
    setSingleColumns(result_dataframe)
    setKColumns(k, result_dataframe)

    for index, row in x.iterrows():
        
        neighbors_ids, neighbors_distance = getNearestNeighbours(neigh, index, row)
        neighbors_types = getNeighbourTypesById(y, neighbors_ids)
        row_type = y[index]
        
        oppositeClassCount(index, neighbors_types, result_dataframe, row_type, k)
        sameClassCount(index, neighbors_types, result_dataframe, row_type, k)
        meanDistanceFromAny(index, neighbors_distance, k, result_dataframe)
        meanDistanceFromSame(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, k, result_dataframe)
        smallestDistanceSameClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dataframe)
        smallestDistanceOppositeClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dataframe)
        smallestDistanceAnyClass(neighbors_distance, index, result_dataframe)
    
    #smallestDistanceSameClass(x, y, k, result_dataframe)
   # smallestDistanceAnyClass(x, y, k, result_dataframe)
   

s = "test"