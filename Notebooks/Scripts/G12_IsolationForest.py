#To use this in a Jupyter Notebook use this line - from G12_IsolationForest import iforestmethod

#Imports IsolationForest() and train_test_split()
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# mainmethod does everything
# pd_csv: a csv read through panda - THIS IS NECESSARY
# custom_random_state: chooses its random seed. You can edit it, but it defaults to 42. 
    #NOTE: This effects the random seed of the IsolationForest() and train_test_split()
# custom_test_size: Choose test_size for train_test_split(), default is 0.2
# custom_train_size: Choose train_size for train_test_split(), default is 0.8

#Returns:
    # train_anomaly_scores: a float for every row in the training set.
        #   Represents how "normal" a transaction is — more negative = more anomalous, higher = more normal.
        #   Gives a continuous signal a downstream supervised model can use as a feature.
    # train_anomaly_labels: an integer for every row in the training set.
        #   Binary: -1 = anomaly, 1 = normal.
        #   A simple categorical flag a supervised model can consume directly as an additional feature. 
    # test_anomaly_scores: same as train_anomaly_scores but for the held-out test set.
        #   The forest was never trained on this data, so these scores reflect how the model generalizes.
        #   Needed to attach anomaly features to unseen data without leakage.
    # test_anomaly_labels: same as train_anomaly_labels but for the held-out test set.
    #   Needed so the supervised model has anomaly labels for test rows too.
def iforestmethod(pd_csv, custom_random_state=42, custom_test_size=0.2, custom_train_size=0.8):
    #Create Isolation Forest
    i_forest = IsolationForest(random_state = custom_random_state)
    
    # Selects only the columns which are numeric - we are basically dropping the class column
    numeric_csv = pd_csv.drop('Class',axis=1)

    #Create Train/Test Split
    train, test = train_test_split(numeric_csv, 
                                   test_size=custom_test_size, 
                                   train_size=custom_train_size, 
                                   random_state=custom_random_state)

    # Fit on train only (no leakage)
    i_forest.fit(train)
    
    # Anomaly scores: more negative = more anomalous
    # Anomaly labels: -1 = anomaly, 1 = normal
    train_anomaly_scores = i_forest.decision_function(train)
    train_anomaly_labels = i_forest.predict(train)

    # Apply to unseen test data — captures new/unseen fraud patterns
    test_anomaly_scores = i_forest.decision_function(test)
    test_anomaly_labels = i_forest.predict(test)
        
    return train_anomaly_scores, train_anomaly_labels, test_anomaly_scores, test_anomaly_labels