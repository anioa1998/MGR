import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def prepareData(dateframe):
    x = pd.DataFrame(dateframe.iloc[:, 0:4])
    y = pd.DataFrame(dateframe.iloc[:,4])

    # skonwertuj kategorie na liczby
    y = LabelEncoder().fit_transform(y)

    # skonwertuj długości nma zakresy 0.0 - 1.0
    x = MinMaxScaler().fit_transform(x)

    # skonwertuj typ z ndarray na DataFrame (wymagane przez klasyfikatory)
    x = pd.DataFrame.from_records(x)
    s = 'test'


def oppositeClassCount(data, k):
    neigh = NearestNeighbors(n_neighbors=k)
    
    for index, row in data.iterrows():
        #Initialization
        type_list = []
        
        #Prepare data
        test_row = row.drop("Species").to_frame().transpose()
        test_dateframe = data.drop(index)
        test_x = test_dateframe.drop("Species", axis=1)
        test_y = list(test_dateframe.iloc[:,-1])
        
        #Get neighbors
        neigh.fit(test_x,test_y)
        neighbors_ids = neigh.kneighbors(test_row,return_distance=False)
        
        #Get type of neighbors
        for i in neighbors_ids:
            type_list.append(test_y[i.astype(int)]) 
  
        

irisset = pd.read_csv(
	filepath_or_buffer='C:\\Users\\Ania\\Desktop\\_STUDIA\\MGR\\iris.csv',
	sep=',',
	encoding='utf-8'
)
irisset = irisset.set_index('Id')
prepareData(irisset)

#oppositeClassCount(irisset,3)


#x = iris.data
#y = iris.target
#knn = KNeighborsClassifier(n_neighbors = 6)

#knn.fit(x,y)
#yp = knn.predict(x)
#acc = metrics.accuracy_score(y, yp)
#print(f"Dokladnosc {acc}")

#samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
#from sklearn.neighbors import NearestNeighbors
#neigh = NearestNeighbors(n_neighbors=1)
#neigh.fit(samples)
#NearestNeighbors(n_neighbors=1)
#print(neigh.kneighbors([[1., 1., 1.]]))
#issue test

	