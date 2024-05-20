from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def select_data(database):
    training_data = database.sample(n=100, random_state=25)
    left_data = database.drop(training_data.index)
    
    training_data.reset_index(drop=True, inplace=True)
    left_data.reset_index(drop=True, inplace=True)
    
    return training_data, left_data


def knn_classifier(training_data, data, k):
    #Calculate the distance between the data and each row of training data
    training_data_distance = pd.DataFrame(columns=['distance', 'class'])
    for training_data_index in training_data.index:
        distance = np.sqrt(np.sum((training_data.iloc[training_data_index,:-1] - data)**2))
        #get class
        clas = training_data.iloc[training_data_index,-1]
        ##add to training_data_distance
        data_add = pd.DataFrame({'distance': [distance], 'class': [clas] })
        training_data_distance = pd.concat([training_data_distance, data_add], ignore_index=True)
    
    training_data_distance.sort_values('distance', inplace=True)
    ##Select the k nearest neighbors
    training_data_distance = training_data_distance.head(k)
    ##Count the number of each class in the k nearest neighbors
    count = training_data_distance['class'].value_counts()
    ##Return the class with the highest count
    return count.idxmax()
    
    
    
    
def main():
    ##Load from xls
    iris_database = pd.read_excel('iris.xls')
    ##Select the data
    iris = iris_database[['Petal_length','Sepal_width','Sepal_length', 'Species_name']]
    training_data, left_data = select_data(iris)
    ##Data frame of the data classified
    data_classified = pd.DataFrame(columns=['Petal_length', 'Sepal_width', 'Sepal_length', 'Species_name'])
    #Data classified
    for index in left_data.index:
        data_to_c = left_data.iloc[index,:-1]
        classified_class = knn_classifier(training_data, data_to_c, 5)
        data_csf = pd.DataFrame({'Petal_length': [data_to_c[0]], 'Sepal_width': [data_to_c[1]], 'Sepal_length': [data_to_c[2]], 'Species_name': [classified_class]})
        data_classified = pd.concat([data_classified, data_csf], ignore_index=True)
    ##Calculate the accuracy
    ##Get the data classified correctly
    match = pd.DataFrame()
    # Iterar sobre los índices de data_classified
    for i in range(len(data_classified)):
        clas1 = data_classified.iloc[i]['Species_name']
        clas2 = left_data.iloc[i]['Species_name']
        if clas1 == clas2:
            match = pd.concat([match, pd.DataFrame({'Petal_length': [data_classified.iloc[i]['Petal_length']], 'Sepal_width': [data_classified.iloc[i]['Sepal_width']], 'Sepal_length': [data_classified.iloc[i]['Sepal_length']], 'Species_name': [clas1]})], ignore_index=True)
    plot(left_data, data_classified, match)
    
    

    
def plot(original_data, classified_data, match_data): 
    ##Extract information from class
    selected_setosa = original_data[original_data['Species_name'].str.strip() == 'Setosa']
    selected_verginica = original_data[original_data['Species_name'].str.strip() == 'Verginica']
    selected_versicolor = original_data[original_data['Species_name'].str.strip() == 'Versicolor']

    selected_setosa_classified = classified_data[classified_data['Species_name'].str.strip() == 'Setosa']
    selected_verginica_classified = classified_data[classified_data['Species_name'].str.strip() == 'Verginica']
    selected_versicolor_classified = classified_data[classified_data['Species_name'].str.strip() == 'Versicolor']
    
    match_setosa = match_data[match_data['Species_name'].str.strip() == 'Setosa']
    match_verginica = match_data[match_data['Species_name'].str.strip() == 'Verginica']
    match_versicolor = match_data[match_data['Species_name'].str.strip() == 'Versicolor']
    
       
    # Crear una figura y tres ejes para los gráficos 3D
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    # Plotear los datos clasificados en el primer subgráfico
    ax1.scatter3D(selected_setosa['Petal_length'], selected_setosa['Sepal_width'], selected_setosa['Sepal_length'], label='classified - Setosa')
    ax1.scatter3D(selected_verginica['Petal_length'], selected_verginica['Sepal_width'], selected_verginica['Sepal_length'], label='classified - Verginica')
    ax1.scatter3D(selected_versicolor['Petal_length'], selected_versicolor['Sepal_width'], selected_versicolor['Sepal_length'], label='classified - Versicolor')
    ax1.set_xlabel('Petal length')
    ax1.set_ylabel('Sepal width')
    ax1.set_zlabel('Sepal length')
    ax1.legend()

    # Plotear los datos originales en el segundo subgráfico
    ax2.scatter3D(selected_setosa_classified['Petal_length'], selected_setosa_classified['Sepal_width'], selected_setosa_classified['Sepal_length'], label='original - Setosa') 
    ax2.scatter3D(selected_verginica_classified['Petal_length'], selected_verginica_classified['Sepal_width'], selected_verginica_classified['Sepal_length'], label='original - Verginica')
    ax2.scatter3D(selected_versicolor_classified['Petal_length'], selected_versicolor_classified['Sepal_width'], selected_versicolor_classified['Sepal_length'], label='original - Versicolor')
    ax2.set_xlabel('Petal length')
    ax2.set_ylabel('Sepal width')
    ax2.set_zlabel('Sepal length')
    ax2.legend()

    # Plotear los datos coincidentes en el tercer subgráfico
    ax3.scatter3D(match_setosa['Petal_length'], match_setosa['Sepal_width'], match_setosa['Sepal_length'], label='match - Setosa')
    ax3.scatter3D(match_verginica['Petal_length'], match_verginica['Sepal_width'], match_verginica['Sepal_length'], label='match - Verginica')
    ax3.scatter3D(match_versicolor['Petal_length'], match_versicolor['Sepal_width'], match_versicolor['Sepal_length'], label='match - Versicolor')
    ax3.set_xlabel('Petal length')
    ax3.set_ylabel('Sepal width')
    ax3.set_zlabel('Sepal length')
    ax3.legend()

    # Ajustar el espacio entre los subgráficos
    plt.tight_layout()

    # Mostrar los subgráficos
    plt.show()
        
    
main()