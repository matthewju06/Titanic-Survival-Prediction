import pandas as pd #Imports/Reads Titanic dataset csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv('/Users/matthewju/Developer/Titanic Survival Prediction/titanic_data.csv')

# Prepare feature (X) and target (y) data
X = titanic_data[['Age', 'Sex', 'sibsp', 'Pclass']] #Input Values
y = titanic_data['2urvived'] #Output Value 0 = Dead, 1 = Alive

#Machine Learning Models
#Essentially we are using our data on 5 different machine learning algorithms to see which one can make the most accurate predictions

def models():
    
    #Support Vector Machines
    from sklearn import svm #Importing model
    svm = svm.SVC() #Setting up each model
    svm.fit(X_train, y_train) #Setting data to model
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression()
    log.fit(X_train, y_train)
    
    #K Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, y_train)
    
    print('(1) SVM Accuracy:', accuracy_score(y_test, svm.predict(X_test))) #Getting % accuracy of the model when comparing
    print('(2) Decision Tree Accuracy:', accuracy_score(y_test, tree.predict(X_test)))
    print('(3) Logistic Regression Accuracy:', accuracy_score(y_test, log.predict(X_test)))
    print('(4) KNN Accuracy:', accuracy_score(y_test, knn.predict(X_test)))
    print('(5) Random Forest Accuracy:', accuracy_score(y_test, forest.predict(X_test)))
    
    return

#Splitting the whole dataset into training and testing data. X is input, y is output (survival)
#Convention is to have 80% of data be used to train the ML algorithms and 20% to test the trained models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
models()

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train.values, y_train.values)

if log.predict([[float(input('Age:')), int(input('Sex (0=M, 1=F):')), int(input('Siblings/Spouse:')), int(input('Class (1-3)'))]]) == 1:
    print("You survived the Titanic!")
else:
    print("You did not survive the Titanic!")