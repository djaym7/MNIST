from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#import data
mnist = fetch_mldata("MNIST original")
X,y = mnist["data"],mnist["target"]

#visualize data/ understand
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation = "nearest")
plt.axis("off")
plt.show()

#test and train split
X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]


#reshuffle index
shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#BUilding model     Stochastic gradient classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train,y_train_5)
#Checking accuracy of classifier using crossvalidation
from sklearn.model_selection import cross_val_predict
y_train_predict = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
y_scores =  cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')

#CHECKING USING CONFUSION MATRIx
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_train_5,y_train_predict)
#check f-1 score
from sklearn.metrics import f1_score
print(f1_score(y_train_5,y_train_predict)
# roc curve
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=None)
#    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
plot_roc_curve(fpr,tpr)
plt.show()
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    