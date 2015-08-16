#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy as np

### Function to Plot graph to visualize the data points
def plotGraph(data1,Label1, Label2, indx1=None, indx2=None, indx3=None):
    plt = matplotlib.pyplot
    for point in data1:    
        if indx2 == None:
            plt.scatter( point[indx1])        
        elif indx3 == None:
            plt.scatter( point[indx1], point[indx2])         
        else:
            if point[indx3] == 1:
                plt.scatter( point[indx1], point[indx2], color = "r", label = "poi")        
            else:
                plt.scatter( point[indx1], point[indx2], color = "b", label = "non-poi")        
    if indx2 == None:        
        plt.xlabel(Label1)
    else:
        plt.xlabel(Label1)
        plt.ylabel(Label2)
    plt.show()        

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Step 1: Identify Outliers
data = featureFormat(data_dict, features_list)
plotGraph(data,"Total_Stock_Value", "total_payments", 8,3,0)
### Step 2: Remove Outliers
### Remove the Outlier "TOTAL"
data_dict.pop("TOTAL",0) 
### Remove the data point "THE TRAVEL AGENCY IN THE PARK" as it has only the "Other" 
### and "Total Payments" populated and rest of the fields are not available
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0) 

### Task 3: Create new feature(s)
for key in data_dict:
    ### New Feature for Shared receipt as a fraction of total messages received     
    fraction_shared_receipt_with_poi = float(data_dict[key]['shared_receipt_with_poi']) / float(data_dict[key]['to_messages'])    
    if np.isnan(fraction_shared_receipt_with_poi):     
        data_dict[key]['fraction_shared_receipt_with_poi'] =  0
    else:
        data_dict[key]['fraction_shared_receipt_with_poi'] =  round(fraction_shared_receipt_with_poi,2)
    ### New Feature for fraction of messages received from poi     
    fraction_from_poi = float(data_dict[key]['from_poi_to_this_person']) / float(data_dict[key]['to_messages']) 
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_from_poi'] =  0
    else:
        data_dict[key]['fraction_from_poi'] =  round(fraction_from_poi,2)
     ### New Feature for fraction of messages sent to poi
    fraction_to_poi = float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_to_poi'] =  0
    else:
        data_dict[key]['fraction_to_poi'] =  round(fraction_to_poi,2)
    
### Select K-Best features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
selector=SelectKBest(f_classif,k=6)
selector.fit(features,labels)
### print the 6 best features and their indices
print "The Best 6 features to be used for the Classification is as below: "
for i in list(selector.get_support(indices=True)):
    print features_list[i + 1], ' - Feature Index: ', i + 1
print " "    

### Plotting the 6 Best features identified in the above step
plotGraph(data,"Salary","Bonus", 1, 5, 0)
plotGraph(data,"Deferred Income","Total Stock Value", 7, 8, 0)
plotGraph(data,"Exercised Stock Options","Fraction to POI", 10, 22, 0)
### Revised feature list containing the 6 best features having the discriminating power
features_list = ['poi','salary', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'fraction_to_poi']
### Final feature list giving the best results for the classifier 
features_list = ['poi','salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'fraction_to_poi','fraction_shared_receipt_with_poi']
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Uncomment to set the algorithm of the classifier to Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Uncomment to set the algorithm of the classifier to SVC
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.svm import SVC
#from sklearn.pipeline import Pipeline
#estimators = [('scale', MinMaxScaler()), ('SVC', SVC(kernel='linear', gamma=5.0, C=1000))]
#clf = Pipeline(estimators)

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)