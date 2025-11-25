import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart_cleveland.csv")




data.head()



print(data.shape)
print(data.dtypes)




X = data.filter(['age','trestbps','chol','thalach','oldpeak'])
X.head()


Y = data['condition'].to_numpy()
# ou
# Y = (data['condition'].values)
# Y = (data.filter(['condition']).values)[:,0]
# Y = (np.array(data.filter(['condition'])).T)[0]
print(Y)
Y.shape
type(Y)



Y.mean()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)




from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)




pred = knn_model.predict(X_test)
pred




print(y_train.mean())
print(y_test.mean())




(pred != y_test).mean()


np.arange(1,81,2)

err = []
for k in np.arange(1,81,2):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    Y_pred = knn_model.predict(X_test)
    err.append((Y_pred != y_test).mean())



plt.figure()
plt.plot(np.arange(1,81,2), err, '.-')
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification')
plt.grid(alpha=0.3)
plt.show()

np.arange(1,81,2)[np.argmin(np.array(err))]


from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": range(1, 81,2)}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=10)
gridsearch.fit(X_train, y_train)

gridsearch.best_params_

gridsearch.cv_results_['mean_test_score']

plt.figure()
plt.plot(np.arange(1,81,2), 1-gridsearch.cv_results_['mean_test_score'], '.-')
plt.grid(alpha=0.3)
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification (CV)')
plt.show()



from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

np.mean(X_train_standard)




err = []
for k in np.arange(1,81,2):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_standard, y_train)
    Y_pred = knn_model.predict(X_test_standard)
    err.append((Y_pred != y_test).mean())



plt.figure()
plt.plot(np.arange(1,81,2), err, '.-')
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification')
plt.grid(alpha=0.3)
plt.show()




np.arange(1,81,2)[np.argmin(np.array(err))]




from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": range(1, 81,2)}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=20)
gridsearch.fit(X_train_standard, y_train)




gridsearch.best_params_
gridsearch.best_score_
gridsearch.cv_results_['mean_test_score']



plt.figure()
plt.plot(np.arange(1,81,2), 1-gridsearch.cv_results_['mean_test_score'])
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification (CV)')
plt.grid(alpha=0.3)
plt.show()


X_standard = standard_scaler.fit_transform(X)
X_standard = pd.DataFrame(X_standard)

X.columns
X_standard.columns = X.columns

newcol = {'cp' : data['cp'], 'exang' : data['exang'],'slope' : data['slope'],'ca' : data['ca'],'thal' : data['thal']}
X_new = X_standard.assign(**newcol)
X_new

X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, Y, test_size=0.2)
print(X_train_new.shape)
print(X_test_new.shape)



from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=18)
knn_model.fit(X_train_new, y_train)




from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": range(1, 81,2)}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=20)
gridsearch.fit(X_train_new, y_train)



plt.figure()
plt.plot(np.arange(1,81,2), 1-gridsearch.cv_results_['mean_test_score'])
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification (CV)')
plt.grid(alpha=0.3)
plt.show()

##Try to change the kernel##
from sklearn.model_selection import GridSearchCV
weights = ['uniform', 'distance']
parameters = {"n_neighbors": range(1, 81,2), "weights": weights}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=10)
gridsearch.fit(X_train_new, y_train)


gridsearch.best_params_
gridsearch.best_score_
gridsearch.cv_results_['mean_test_score']

gridsearch.cv_results_['params']

results = pd.DataFrame(gridsearch.cv_results_)
print(results)
results.columns

results_short = results.pivot(index='param_n_neighbors', columns='param_weights', values='mean_test_score')
results_short

plt.figure()
plt.plot(np.arange(1,81,2), 1-results_short['distance'],'r.-')
plt.plot(np.arange(1,81,2), 1-results_short['uniform'],'b.-')
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification (CV)')
plt.grid(alpha=0.3)
plt.legend(['distance','uniform'])
plt.show()

##Gaussian kernel
def gaussian_kernel(distances): 
    weights = np.exp(- (distances ** 2) / 2)
    return weights / np.sum(weights)
# (ex) gaussian_kernel(np.array([0.5, 1.0, 1.5]))

parameters = {"n_neighbors": range(1, 121,2), "weights": [gaussian_kernel]}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=10)
gridsearch.fit(X_train_new, y_train)


gridsearch.best_params_
1-gridsearch.best_score_

plt.figure()
plt.plot(np.arange(1,121,2), 1-gridsearch.cv_results_['mean_test_score'])
plt.xlabel('k')
plt.ylabel('Taux de mauvaise classification (CV)')
plt.grid(alpha=0.3)
plt.show()

#Try to optimise with respect to many parameters
weights = ['uniform', 'distance',gaussian_kernel]
parameters = {"n_neighbors": range(1, 121,2), "weights": weights, "p": [0.5,1,2], "algorithm": ['brute']}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,cv=5)
gridsearch.fit(X_train_new, y_train)

gridsearch.best_params_
1-gridsearch.best_score_

# Pour terminer, choisir le modele avec les parametres optimaux et evaluer sur le test set
# knn_model = KNeighborsClassifier(n_neighbors=, weights='', p=)
knn_model.fit(X_train_new, y_train)
pred = knn_model.predict(X_test_new)
(pred != y_test).mean()

