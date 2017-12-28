import numpy as np
from svmutil import *

# y, x = svm_read_problem("/Users/islab/Downloads/libsvm-3.22/heart_scale")
# m = svm_train(y[:200], x[:200], '-c 4')
# p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)


if __name__ == "__main__":

    # x_train = np.loadtxt("X_train.csv", delimiter=',')
    y_train = np.loadtxt("T_train.csv", dtype=int)
    # x_test = np.loadtxt("X_test.csv", delimiter=',')
    y_test = np.loadtxt("T_test.csv", dtype=int)

    train_data = open("X_train.data", 'w+')
    test_data = open("X_test.data", 'w+')

    for i, line in enumerate(train_data):
        line[0] = y_train[i].__str__

    print(y_train[0].__str__())