import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from utility import make_meshgrid, plot_contours, MidpointNormalize
from pca import pca


print("Loading data...")
x_train = np.loadtxt("data/X_train.csv", delimiter=',')
x_test = np.loadtxt("data/X_test.csv", delimiter=',')
y_train = np.loadtxt("data/T_train.csv", dtype=int)
y_test = np.loadtxt("data/T_test.csv", dtype=int)


def draw_heatmap():
    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.savefig("grid_search_heatmap.png")


# SVM
print("Classifying...")

# Remember to uncomment the following lines to do grid search
# print("Grid searching")
# C_range = np.logspace(-2, 6, 5, base=2)
# gamma_range = np.logspace(-7, 0, 5, base=2)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=2)
# grid.fit(x_train, y_train)
# print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
# scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
# draw_heatmap()

print("Training...")
# Optimal C=4, gamma=0.026780129
# clf = svm.SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
clf = svm.SVC(C=4, gamma=0.026780129)
clf.fit(x_train, y_train)
joblib.dump(clf, 'classifier.pkl')

# Load trained model
clf = joblib.load('classifier.pkl')

print("Testing...")
acc = clf.score(x_test, y_test)
print("    Accuracy:", acc)

# PCA
print("PCA...")
# pca = PCA(n_components=2, copy=False, whiten=False)
# embedded_data = pca.fit_transform(x_train)
embedded_data, W = pca(x_train, k=2)

# title for the plots
titles = ('SVC with RBF kernel and then PCA',
          'Support vectors',
          'Decision boundary')

color = ['y', 'm', 'c', 'b', 'g']

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

print("Drawing PCA results...")
for i in range(embedded_data.shape[0]):
    ax1.scatter(embedded_data[i, 0], embedded_data[i, 1], c=color[y_train[i]-1], alpha=0.5, s=1)
ax1.set_title(titles[0])

print("Drawing support vectors...")
sv_indicator = np.zeros(embedded_data.shape[0], dtype=int)
sv_indicator[clf.support_] = 1
mark = ['o', 'D']
for i in range(embedded_data.shape[0]):
    if sv_indicator[i] == 0:
        ax2.scatter(embedded_data[i, 0], embedded_data[i, 1],
                    c=color[y_train[i]-1], alpha=0.2, s=1, marker=mark[sv_indicator[i]])
    else:
        ax2.scatter(embedded_data[i, 0], embedded_data[i, 1],
                    c=color[y_train[i]-1], alpha=1, s=2, marker=mark[sv_indicator[i]])
ax2.set_title(titles[1])


clf = svm.SVC(C=4, gamma=0.03125)
clf.fit(embedded_data, y_train)

print("Drawing decision boundary...")
for i in range(embedded_data.shape[0]):
    ax3.scatter(embedded_data[i, 0], embedded_data[i, 1], c=color[y_train[i]-1], alpha=0.5, s=1)
xx, yy = make_meshgrid(embedded_data[:, 0], embedded_data[:, 1])
plot_contours(ax3, clf, xx, yy, np.transpose(W), cmap=plt.cm.coolwarm, alpha=0.8)
ax3.set_title(titles[2])

gs.tight_layout(fig)
plt.savefig("result.png")
plt.show()
