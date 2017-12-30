import numpy as np
from svmutil import *
from grid import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_2ddata(data, label):
    num_data, dim = data.shape
    if dim != 2:
        raise Exception("Sorry, I can only visualize 2D data.")

    # Draw data
    plt.scatter(data[:, 0], data[:, 1], c=label, label=label, alpha=0.5, s=1)
    plt.show()


if __name__ == "__main__":
    # y_train, x_train = svm_read_problem("train.data")

    # Grid search for optimal parameters C and Gamma
    # rate, param = find_parameters('train.data', '-log2c -1,2,1 -log2g -5,1,1')
    # c = param['c']
    # g = param['g']

    # Optimal
    c = 4
    g = 0.03125

    print("Parameters:", "c=%f," % c, "g=%f" % g)

    # Train model
    #    -t: kernel_type (2: RBF), -s: svm_type (0: C_-SVC)
    # train_option = '-c ' + c.__str__() + ' -g ' + g.__str__() + ' -t 2 -s 0'
    # m = svm_train(y_train, x_train, train_option)
    # svm_save_model('mnist.model', m)

    # Test model
    m = svm_load_model('mnist.model')
    y_test_libsvm_format, x_test_libsvm_format = svm_read_problem("test.data")
    p_label, p_acc, p_val = svm_predict(y_test_libsvm_format, x_test_libsvm_format, m)

    x_test = np.loadtxt("X_test.csv", delimiter=',')

    # PCA
    pca = PCA(n_components=2, copy=False, whiten=False)
    embedded_data = pca.fit_transform(x_test)
    visualize_2ddata(embedded_data, p_label)

    print(m)