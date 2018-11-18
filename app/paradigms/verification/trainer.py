def train(classifier, train_data, train_label, is_neural_net=False):
    
    # fit the training dataset on the classifier
    trained_classifier = classifier.fit(train_data, train_label)

    return trained_classifier