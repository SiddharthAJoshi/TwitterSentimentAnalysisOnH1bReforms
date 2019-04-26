#classification.py
import pandas as pd
import urllib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier

# loading data from csv file using pandas
data = pd.read_csv('analysis21.csv')
'''
 Similarly, max_df specifies that only use those words that occur in a maximum of 80% of the documents.
 Words that occur in all documents are too common and are not very useful for classification.
 Similarly, min-df is set to 7 which shows that include words that occur in at least 7 documents.
'''
# min_df means discard words appearing in less than two tweets
# max_df means discard words appearing in more than 90% of the tweets


vectorizer = TfidfVectorizer(min_df=5,max_df=0.8,lowercase="True",stop_words="english")

X = vectorizer.fit_transform(data.Tweet.values.astype('U'))
y=data.Sentiment

data['X']=list(X)
data['y']=data.Sentiment

'''
we use the train_test_split class from the sklearn.model_selection module to divide our data 
into training and testing set. The method takes the feature set as the first parameter, the 
label set as the second parameter, and a value for the test_size parameter. We specified a 
value of 0.2 for test_size which means that our data set will be split into two sets of 80%
 and 20% data. We will use the 80% dataset for training and 20% dataset for testing. 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
-------------------------------------------------------------------------------------------------
Using Multinomial Naive Baiyes algorithm to cllassify and predict
-------------------------------------------------------------------------------------------------
'''
clf = MultinomialNB()
clf.fit(X_train,y_train)

'''
Once the model has been trained, the last step is to make predictions on the model. To do so,
we need to call the predict method on the object of the Multinomial NB class that we 
used for training
'''
predictions = clf.predict(X_test)
'''
To find the values for these metrics, we can use classification_report, confusion_matrix,
and accuracy_score utilities from the sklearn.metrics library.
'''
print('Multinomial Naive Bayes Result')
print('The confusion matrix')
print(confusion_matrix(y_test,predictions))
print('The classification report')
print(classification_report(y_test,predictions))
print('The accuaracy score')
print(accuracy_score(y_test, predictions))
'''
-------------------------------------------------------------------------------------------------
Plot confustion matrix.
-------------------------------------------------------------------------------------------------
'''

nb_true = y_test
skplt.metrics.plot_confusion_matrix(nb_true, predictions)
plt.show()
#skplt.estimators.plot_feature_importances(clf)
skplt.estimators.plot_learning_curve(clf, X, y)
plt.show()
series=pd.Series(predictions)

data["NaiveBayes"]=series

'''
-------------------------------------------------------------------------------------------------
Using SVM algorithm to cllassify and predict
-------------------------------------------------------------------------------------------------
'''
clf1 = svm.SVC(kernel="linear", verbose=3)
clf1.fit(X_train,y_train)
'''
Once the model has been trained, the last step is to make predictions on the model. To do so,
we need to call the predict method on the object of the SVM class that we 
used for training
'''
predictions1 = clf1.predict(X_test)

'''
To find the values for these metrics, we can use classification_report, confusion_matrix,
and accuracy_score utilities from the sklearn.metrics library.
'''
print('-------------------------------------------------------------------------------------------------')
print('SVM Result')
print('The confusion matrix')
print(confusion_matrix(y_test,predictions1))
print('The classification report')
print(classification_report(y_test,predictions1))
print('The accuaracy score')
print(accuracy_score(y_test, predictions1))
series1=pd.Series(predictions1)
'''
-------------------------------------------------------------------------------------------------
Plot confustion matrix and learning curve.
-------------------------------------------------------------------------------------------------
'''
svm_true = y_test
skplt.metrics.plot_confusion_matrix(svm_true, predictions1)
plt.show()
skplt.estimators.plot_learning_curve(clf1, X, y)
plt.show()
data["SVM"]=series1


'''
-------------------------------------------------------------------------------------------------
Using Random Forest Classifier algorithm to cllassify and predict
-------------------------------------------------------------------------------------------------
'''
clf2 = RandomForestClassifier(n_estimators=200, random_state=0) 
clf2.fit(X_train,y_train)
'''
Once the model has been trained, the last step is to make predictions on the model. To do so,
we need to call the predict method on the object of the RandomForestClassifier class that we 
used for training
'''
predictions2 = clf2.predict(X_test)
'''
To find the values for these metrics, we can use classification_report, confusion_matrix,
and accuracy_score utilities from the sklearn.metrics library.
'''
print('-------------------------------------------------------------------------------------------------')
print('Random Forest Classifier Result')
print('The confusion matrix')
print(confusion_matrix(y_test,predictions2))
print('The classification report')
print(classification_report(y_test,predictions2))
print('The accuaracy score')
print(accuracy_score(y_test, predictions2))
series2=pd.Series(predictions2)
'''
-------------------------------------------------------------------------------------------------
Plot confustion matrix.
-------------------------------------------------------------------------------------------------
'''
rf_true = y_test
skplt.metrics.plot_confusion_matrix(rf_true, predictions2)
plt.show()
skplt.estimators.plot_learning_curve(clf2, X, y)
plt.show()
data["RandomForest"]=series2



data.to_csv('NResult.csv')
