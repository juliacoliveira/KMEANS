from scipy.stats import mode
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk
from scipy.stats import zscore
import numpy as np
from sklearn.cluster import KMeans

# Opening the file
csv_file = 'C:/Users/julia/Downloads/7 semestre/IA/Projeto/KMEANS/diamonds_training.csv' #need to change the path of the file csv
df = pd.read_csv(csv_file, delimiter=';')
print (df.head(3))

indexNames = df[ df['price'] == 4].index
df = df.drop(indexNames)
indexNames = df[ df['price'] == 3].index
df = df.drop(indexNames)

# Getting the columns names
linear_vars = df.select_dtypes(include = [np.number]).columns
linear_vars =  list(linear_vars)

# Verifying and transforming the lines with zero, EXCEPT the 'color' column
colorless_vars = linear_vars.copy()
colorless_vars.remove('color')
colorless_vars.remove('price')
print (linear_vars)
print (colorless_vars)

# Normalizing the atributes, EXCEPT the 'color'
def removeoutliers(df, colorless_vars, z):
    for var in colorless_vars:
        df1 = df[np.abs(zscore(df[var])) < z]
    return df1

df = removeoutliers(df, colorless_vars, 2)

# converting to log scale
def convertToLog(df, colorless_vars):
    for var in colorless_vars:
        df[var] = np.log(df[var])

convertToLog(df, colorless_vars)
#print (df.head(3))

# Divide the normalized data into targets and attributes
attr_df = df.drop(['price'], axis = 1).values.tolist()

targets_df = df.drop(['carat', 'color', 'table', 'x', 'y', 'z'], axis = 1).values.tolist()

#################################### Test data acquisition ######################################################

# Opening the file
csv_file = 'C:/Users/julia/Downloads/7 semestre/IA/Projeto/KMEANS/diamonds_test.csv' #need to change the path of the file csv
df = pd.read_csv(csv_file, delimiter=';')
df_visual = df.copy()
print (df.head(3))

indexNames = df[ df['price'] == 4].index
df = df.drop(indexNames)
indexNames = df[ df['price'] == 3].index
df = df.drop(indexNames)

# Getting the columns names
linear_vars = df.select_dtypes(include = [np.number]).columns
linear_vars =  list(linear_vars)

df = removeoutliers(df, colorless_vars, 2)
convertToLog(df, colorless_vars)

attr_test = df.drop(['price'], axis = 1).values.tolist()

targets_test = df.drop(['carat', 'color', 'table', 'x', 'y', 'z'], axis = 1).values.tolist()

################################ End of the test acquisition #######################################################

# Fun????o para retornar NaN se um vetor do qual se deseja extrair a moda est?? vazio.

def conditionalMode():
  return lambda x : np.nan if x == [] else mode(x).mode[0]

# Function that classifies the clusters as the type of diamond more recurrent in each one of them

def classifyCluster(clf, data, targets):
  predictions = clf.predict(data)
  enumerated_predictions = list(enumerate(predictions))
  elements_clusters = []
  categorized_clusters = []
  mode_fun = conditionalMode() 
  for n in range(0, clf.n_clusters): # Assigns one class to each cluster 
    indexes_cluster = list(map(lambda x : x[0], filter(lambda x : x[1] == n, enumerated_predictions)))
    elements_clusters.append(list(map(lambda i : targets[i], indexes_cluster)))
  categorized_clusters = list(map(mode_fun, elements_clusters))
  return [categorized_clusters, enumerated_predictions]

# Transform the predictions from clusters to classes

def convertPredictions(categorized_clusters, enumed_predictions): 
  realPredictions = list(map(lambda i : categorized_clusters[i[1]], enumed_predictions))
  return realPredictions

# Calculate the Score 

def scoreKMeans(predictions, targets):
  got_right = list(map(lambda x : x[1] == targets[x[0]], enumerate(predictions)))
  score = sum(got_right)/len(got_right)
  return score

rede = KMeans(n_clusters=30, n_init=10)
rede.fit(attr_df)
[catergorized_clusters, enumed_predictions] = classifyCluster(rede, attr_test, targets_test)
converted_predictions = convertPredictions(catergorized_clusters, enumed_predictions)
score = scoreKMeans(converted_predictions,targets_test)

print("Score = {}".format(score))
print("Inercia = {}".format(rede.inertia_))
print("Matriz de confus??o =\n{}\n".format(confusion_matrix(converted_predictions, targets_test)))

######## Trying to plot the graphics ################## 

# Filter rows from different data
#print (enumed_predictions)

clusters_pertencentes = list(map(lambda x: x[1], enumed_predictions))
classes_pertencentes = list(map(lambda x: catergorized_clusters[x][0], clusters_pertencentes))

labels_0 = [x == 0 for x in classes_pertencentes]
labels_1 = [x == 1 for x in classes_pertencentes]
labels_2 = [x == 2 for x in classes_pertencentes]

labels_original_0 = [x == 0 for x in df['price']]
labels_original_1 = [x == 1 for x in df['price']]
labels_original_2 = [x == 2 for x in df['price']]

testes_classe_0 = df[labels_0]
testes_original_0 = df[labels_original_0]

testes_classe_1 = df[labels_1]
testes_original_1 = df[labels_original_1]

testes_classe_2 = df[labels_2]
testes_original_2 = df[labels_original_2]

testes_classes = [testes_classe_0, testes_classe_1, testes_classe_2]
testes_original = [testes_original_0, testes_original_1 , testes_original_2]

carac = ['carat', 'color', 'table', 'x', 'y','z']
carac2 = ["carat", "color", "table", "x", "y","z"]

for z in range (0, 3):

  for y in range (1, 4):

    plt.subplot(3, 3, z*3 + y)

    plt.scatter(testes_classes[y-1][carac[z]], testes_classes[y-1]['price'], marker="*", c="red")
    plt.scatter(testes_original[y-1][carac[z]], testes_original[y-1]['price'], alpha=0.01, c="blue")

    plt.xlabel(carac[z])
    plt.ylabel("price")
plt.show()    

for z in range (0, 3):

  for y in range (1, 4):

    plt.subplot(3, 3, z*3+y)

    plt.scatter(testes_classes[y-1][carac[z+3]], testes_classes[y-1]['price'], marker="*", c="red")
    plt.scatter(testes_original[y-1][carac[z+3]], testes_original[y-1]['price'], alpha=0.01, c="blue")

    plt.xlabel(carac[z+3])
    plt.ylabel("price")

plt.show()



#print (catergorized_clusters[enumed_predictions[0][1]][0])
#clusters = pd.DataFrame[catergorized_clusters]
#filtered_label0 = clusters[label == 0]
#print (filtered_label0)

#plotting the results
'''plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()'''
