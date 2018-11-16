from sklearn import metrics

def train(classifier, train_data, train_label, test_data, test_label, is_neural_net=False):
    
    # fit the training dataset on the classifier
    classifier.fit(train_data, train_label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(test_data)
    
    return metrics.accuracy_score(predictions, test_label)