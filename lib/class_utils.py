#################################################################
# Classification utils library with classification models       #
#   used for document classification using topic definitions    #
#                                                               #
# --> Author: Andres Carrillo Lopez                             #
# --> GitHub: AndresC98@github.com                              #
#                                                               #
# --> Dependencies:                                             #
#      - doc_utils helper library                               #
#      - keras                                                  #
#################################################################


import doc_utils #helper library (Custom)

from keras.models import Sequential
from keras.layers import Dense

def neuralNetClassifier(x_train, y_train, x_test):
    '''
    Given a document topic definition training dataset (x_train, y_train) 
    and a testing data (x_test), returns predicted class.

    Input data needs to be preprocessed with "prepareNeuralNetData()" function.
    '''

    #Neural Architecture Definition
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10000,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    #Model compilation
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    #Model training
    hist = model.fit(x_train, y_train, epochs=5)

    #Output: predicted class of target article
    prediction = model.predict_classes(x_test)

    print("\n======================================RESULT======================================\nPredicted class #:",prediction[0])
    
    return prediction
