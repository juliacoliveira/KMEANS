from scipy.stats import mode
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

#Treinamento dos agrupadores K-Means usando os atributos do conjunto de treino

def trainKMeans(m):
  clf = []
  for n in range(1,m+1):
    clf.append(KMeans(n_clusters=n, n_init=10))
    clf[n-1].fit(attr_df)
  return clf.copy()

# Função para retornar NaN se um vetor do qual se deseja extrair a moda está vazio.

def conditionalMode():
  return lambda x : np.nan if x == [] else mode(x).mode[0]

# Função para obtenção do agrupador KMeans cuja acurácia é máxima e cuja quantidade de clusters é mínima.

def bestKM(sorted_clfs):
  max_accuracy = sorted_clfs[-1][0]
  return min(list(filter(lambda x : x[0] == max_accuracy, sorted_clfs)))

# Função de classificação dos clusters segundo o tipo de flor mais recorrente em cada um deles

def classifyCluster(clf, data, targets):
  predictions = clf.predict(data)
  enumerated_predictions = list(enumerate(predictions))
  elements_clusters = []
  categorized_clusters = []
  mode_fun = conditionalMode()
  for n in range(0, clf.n_clusters):
    indexes_cluster = list(map(lambda x : x[0], filter(lambda x : x[1] == n, enumerated_predictions)))
    elements_clusters.append(list(map(lambda i : targets[i], indexes_cluster)))
  categorized_clusters = list(map(mode_fun, elements_clusters))
  return [categorized_clusters, enumerated_predictions]

def convertPredictions(categorized_clusters, enumed_predictions):
  realPredictions = list(map(lambda i : categorized_clusters[i[1]], enumed_predictions))
  return realPredictions

def scoreKMeans(predictions, targets):
  got_right = list(map(lambda x : x[1] == targets[x[0]], enumerate(predictions)))
  score = sum(got_right)/len(got_right)
  return score

rede = KMeans(n_clusters=69, n_init=10)
rede.fit(attr_df)
[catergorized_clusters, enumed_predictions] = classifyCluster(rede, attr_df, targets_df)
converted_predictions = convertPredictions(catergorized_clusters, enumed_predictions)
score = scoreKMeans(converted_predictions,targets_df)

print("Score = {}".format(score))
print("Inercia = {}".format(rede.inertia_))
print("Matriz de confusão =\n{}\n".format(confusion_matrix(converted_predictions, targets_df)))