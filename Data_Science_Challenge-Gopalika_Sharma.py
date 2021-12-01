#!/usr/bin/env python
# coding: utf-8

# # Chronic Kidney Disease - Identifying risk factors and potential CKD subtypes

# Chronic kidney disease, also called chronic kidney failure, involves a gradual loss of kidney function. Your kidneys filter wastes and excess fluids from your blood, which are then removed in your urine. Advanced chronic kidney disease can cause dangerous levels of fluid, electrolytes and wastes to build up in your body.For this project, Chronic Kidney Disease dataset in UCI Machine learning repository has been explored, which includes 24 attributes excluding the target label class and health parameters of 400 patients.
# 
# 

# ## Getting to know the Data



import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns

#used Weka to convert .arff to .csv

data = pd.read_csv('/Users/gopalika14/Desktop/chronic_kidney_disease.csv',sep=',', na_values=['?'])
display(data.head())
print (" ")
print ("Attribute list is {}".format(data.columns.values))

print (" ")
#No of people with chronic kidney disease
num_ckd = len(data[data['class']=='ckd'])

#No of people without chronic kidney disease
num_notckd = len(data[data['class']=='notckd'])

#Visualising the class
ax=sns.countplot(x="class", data=data)
for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), xy=(p.get_x()+p.get_width()/2, p.get_height()),
            xytext=(0, 5), textcoords='offset points', ha="center", va="center");

print(" ")
print ("Number of people detected with chronic kidney disease: {}".format(num_ckd))
print ("Number of people not detected with chronic kidney diesease: {}".format(num_notckd))


#getting the distribution of the numerical attibutes of the dataset
display(data.describe())


#plotting the heatmap to establish correlation between attributes 
plt.figure(figsize=(30,20))
ax = sns.heatmap(data.corr(), annot=True)
ax.set_title('Feature Correlation',fontsize=10)
ax.yaxis.set_tick_params(labelsize= 16)
ax.xaxis.set_tick_params(labelsize= 16)
plt.show()


# #### OBSERVATION 1: We have 24 feature attributes with 11 numerical attributes and 14 nominal and one target label.The number of people detected with chronic kidney disease(ckd) is 250 and with no-ckd is 150. In the correlation plot, hemo,pcv and rbcc shows high positive correlation with each other followed by su,bgr which are also in the high end of positive correlation.

# ## Pre-Processing the Data

# checking null values within the dataset
data.isnull().sum()


#Replacing missing values with most frequent value for nominal features and median for numerical features
X = pd.DataFrame(data)
fill = pd.Series([X[c].value_counts().index[0]
        if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
        index=X.columns)
clean_data=X.fillna(fill) 
clean_data.head()            


#checking if the null values has been replaced
clean_data.isnull().sum()


#Using Label Encoder to encode nominal values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df=clean_data.copy()
for items in df:
    if df[items].dtype == np.dtype('O'):
        df[items]=le.fit_transform(df[items])

print(df.dtypes)
print(df)



#Let's Scale the features for uniformity
from sklearn.preprocessing import StandardScaler, RobustScaler
target_class = df['class']
features = df.drop('class', axis = 1)
data_robust = pd.DataFrame(RobustScaler().fit_transform(features), columns=features.columns)


# ### In order to identify the risk factors for the CKD I plan to run a classification model so I can identify the risks through feature importance as it helps us identify which features contributed heavily towards the classification of CKD, hence it identifies the risk factors.



# For the purpose of classification I chose Random Forest Model, this can be altered to any preferred model of choice.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(data_robust,target_class, test_size=0.25, random_state=42)



clf=RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())

score = clf.score(X_train, y_train)
print('Train',score) 

score = clf.score(X_test,y_test)
print('Test',score) 

importances = clf.feature_importances_
print(importances)
print(sum(importances))
sort = sorted(importances, reverse=True)
print(sort)




#Testing
y_pred = clf.predict(X_test)
print("Precision score:")
print(round(precision_score(y_test, y_pred, average='binary'), 3))
print("\nAccuracy score:")
print(round(accuracy_score(y_test, y_pred), 4))
#cm = confusion_matrix(y_test, y_pred)
#print(cm)


# ## Identifying Risk factors



# The following feature importance plot shows the features that are heavily inclined towards the classification of CKD, hence it identifies the risk factors. 
rf_fi = pd.DataFrame({'Features':data_robust.columns,'Importance':np.round(clf.feature_importances_,3)}).sort_values('Importance',ascending=True)
rf_fi.plot(x='Features',y='Importance',kind='barh');



#calculating feature importance value
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]



feature_names = data_robust.columns
print(feature_names)




# Feature Ranking represents the value assigned to each feature hence demonstrating their importance towards CKD classification
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))




#Plotting the rankings in a pretty plot
plt.subplots(figsize=(15,8))
sns.barplot(importances, feature_names, palette='inferno')


# ### Observation 2: The main risk factors are identified by plotting the feature importance graph with respect to the classification algorithm as it suggest which features highly influenced the CK disease in a patient. 
# 
# ### The top risk factors are, haemoglobin,packed cell volume,serum cretinine,red blood cells count and sugar as they have high feature importance value.

# ## Identifying Potential CKD subtypes

X = data_robust
y = target_class
print('Shape of X:', X.shape, '\n', 'Shape of y:', y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)



#Running Principal component analysis on all the features to reduce the dimensionality of the large data set, by transforming the variables into a smaller one that still contains most of the information. 
from sklearn.decomposition import PCA
pca = PCA(n_components=24, svd_solver='randomized').fit(X)
print(pca.components_)
print (" ")
print(pca.explained_variance_)




n_components = [5,15,24]
for k in n_components:
    pca = PCA(n_components= k, svd_solver='randomized').fit(X)
    f,ax = plt.subplots()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("ratio of explained variance")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');



#checking for the number of principal components needed to retain 95% of the variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
dimensions = np.argmax(cumsum >= 0.95) + 1
dimensions


# In order to cluster data, we need to determine how to tell if two data points are similar. A proximity measure characterizes the similarity or dissimilarity that exists between objects.
# We can choose to determine if two points are similar. So if the value is large, the points are very similar. Or choose to determine if they are dissimilar. If the value is small, the points are similar. This is what we know as "distance".There are various distances that a clustering algorithm can use: Manhattan distance, Minkowski distance, Euclidean distance, among others.
# 
# 
# K-means typically uses Euclidean distance to determine how similar (or dissimilar) two points are.
# 
# First, we need to fix the numbers of clusters to use.The Elbow method looks at how the total WSS varies with the number of clusters.  For that, we'll compute k-means for a range of different values of k. Then, we calculate the total WSS. We plot the curve WSS vs. number of clusters.  Finally, we locate the elbow or bend of the plot. This point is considered to be the appropriate number of clusters.
# 
# ### For our solution I am running unsupervised clustering on patient's baseline characteristics among 400 participants to identify novel CKD subgroups that best represent the data pattern.



from sklearn.cluster import KMeans
wss = []
#choosing 11 features we got through PCA analysis
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wss.append(km.inertia_)
plt.plot(range(1,11),wss, c="#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wss', size=14)
plt.show() 


# ### The elbow of the plot is 5, this point is considered to be the appropriate number of clusters.


#running k-means on 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=10, n_init=10, random_state=0)
# Fit and predict 
label = kmeans.fit_predict(X)


#Labels for the kmeans cluster
print(label)


# Plotting the first two clusters to demonstrate how the clusters look
import matplotlib.pyplot as plt
pca_2d = pca.transform(X)
#filter rows of original data
filtered_label0 = pca_2d[label == 0]
filtered_label1 = pca_2d[label == 1]
#filtered_label2 = pca_2d[label == 2]
#filtered_label3 = pca_2d[label == 3]
#filtered_label4 = pca_2d[label == 4]

#plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1])
#plt.scatter(filtered_label2[:,0] , filtered_label2[:,1])
#plt.scatter(filtered_label3[:,0] , filtered_label3[:,1])
#plt.scatter(filtered_label4[:,0] , filtered_label4[:,1])
plt.show()


#Getting unique labels
u_labels = np.unique(label)
 
#plotting all the clusters
 
for i in u_labels:
    plt.scatter(pca_2d[label == i , 0] , pca_2d[label == i , 1] , label = i)
plt.legend()
#plt.xlim(-10, 50)
#plt.ylim(-10, 50)
plt.show()


# ### Observation 3: By running unsupervised clustering on patient's baseline characteristics among 400 participants I identified 5 novel CKD subgroups that best represent the data pattern. But cluster marked 0,1 and 3 show more probability of being subgroups due to the high count of point distribution.
# 
# ### Let's identify the centroids in order to find underlying features within clusters so we know what factors are dominating the subtypes of CKD


features = X.columns.tolist()
print(f"Features: \n{features}")

centroids = kmeans.cluster_centers_
print(f"Centroids \n{centroids}")



sorted_centroid_features_idx = centroids.argsort(axis=1)[:,::-1]
print(f"Sorted Feature/Dimension Indexes for Each Centroid in Descending Order: {sorted_centroid_features_idx}")

print()

sorted_centroid_features_values = np.take_along_axis(centroids, sorted_centroid_features_idx, axis=1)
print(f"Sorted Feature/Dimension Values for Each Centroid in Descending Order: {sorted_centroid_features_values}")



#Feature analysis for Cluster marked 0
first_features_in_centroid_1 = centroids[0][sorted_centroid_features_idx[0]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[0]], 
            first_features_in_centroid_1
        )
    ))


# #### Conclusion: Cluster labelled "0" identifies with high valued features like red blood cells, hemoglobin, packed cell volume, sodium.


#Feature analysis for Cluster marked 1
first_features_in_centroid_2 = centroids[1][sorted_centroid_features_idx[1]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[1]], 
            first_features_in_centroid_2
        )
    ))


# #### Conclusion: Cluster labelled "1" identifies with high valued features like serum creatinine, blood urea, albumin, hypertension,diabetes melitus, sugar,anemia, blood pressure, pedal adema, blood glucose random, appetite,potassium,coronary artery disease.


#Feature analysis for Cluster marked 2
first_features_in_centroid_3 = centroids[2][sorted_centroid_features_idx[2]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[2]], 
            first_features_in_centroid_3
        )
    ))


# #### Conclusion: Cluster labelled "2" identifies with high valued features like potassium, serum creatinine, blood urea,sugar, blood glucose random,sodium, albumin,blood pressure,pedal adema, anemia,hypertension,diabetes mellitus, age.


#Feature analysis for Cluster marked 3
first_features_in_centroid_4 = centroids[3][sorted_centroid_features_idx[3]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[3]], 
            first_features_in_centroid_4
        )
    ))


# #### Conclusion: Cluster labelled "3" identifies with high valued features like blood glucose random,sugar,albumin,hypertension,serum creatinine, diabetes mellitus, blood urea, white blood cells count,appetite.


#Feature analysis for Cluster marked 4
first_features_in_centroid_5 = centroids[4][sorted_centroid_features_idx[4]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[4]], 
            first_features_in_centroid_5
        )
    ))


# #### Conclusion: Cluster labelled "4" identifies with high valued features like serum creatinine, blood urea,hypertension, blood glucose random,appetite, coronary artery disease,diabetes and age.






