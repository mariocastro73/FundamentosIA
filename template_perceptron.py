class Perceptron(object):
    """perceptron classifier.
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta = 1.9, n_iter = 100, mean_bias_init = False):
        pass
    
    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
        return self

    def net_input(self, X):
        """Calculate net input before activation"""
        return result # Linear transformation of the input

    def predict(self, X):
        """Return class label after unit step"""
        return activation # Sign function in compact form

