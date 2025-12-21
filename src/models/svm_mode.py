from sklearn.svm import SVC

def build_svm():
    return SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
    # rbf = radial basis function
    # C = regularization parameter
    # gamma = kernel coefficient