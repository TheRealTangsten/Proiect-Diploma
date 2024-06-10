from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import sklearn
import perceptron as perc
import numpy as np

def testScikitPerc(wois = 0, woie = 2048):
    sklearn.set_config(enable_metadata_routing=True)
    clf  = Perceptron(tol=1e-3, random_state=int(1), shuffle=False)
    train_data, train_label = perc.getTrainData(wois, woie)
    #train_data, train_label = prc_at1.getTrainData(wois, woie)
    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm, :]
    train_labels = train_label[perm]
    print(str(train_data.shape) + " " + str(train_labels.shape))

    clf.fit(train_data, train_labels)
    test_data, test_label = perc.getTestData(wois, woie)
    #test_data, test_label = prc_at1.getTestData(wois, woie)
    perm = np.random.permutation(test_data.shape[0])
    test_data = test_data[perm, :]
    test_labels = test_label[perm]
    print(str(test_data.shape) + " " + str(test_labels.shape))
    print(clf.score(test_data,test_labels))
    print("Normal Params: " + str(clf.get_params()))
    print("Verboose Params: " + str(clf.get_params(deep=True)))
    print("Metadata routing: " + str(clf.get_metadata_routing()))
    print("fit_request: " + str(clf.set_fit_request()))
    print("score_request: " + str(clf.set_score_request()))
    print("weights: " + str(clf.coef_))
    print("weights: " + str(clf.coef_.shape))


def testScikitPercN(wois = 0, woie = 2048, iterations = 10):
    for i in range(0, iterations):
        print("Iteration {0}: ".format(i))
        testScikitPerc(wois = wois, woie = woie)
        print("\n\n")
'''
X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
print("number vector" + str(X) + "\nLabel Vector:" + str(y))
clf.fit(X, y)
print(clf.score(X, y))
'''