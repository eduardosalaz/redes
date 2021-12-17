from preprocessing import preprocess
from log import logistic
from mlpclass import mlp_class
from neuraln import deep_nn
from time import perf_counter

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = preprocess()
    # Logistic
    tic = perf_counter()
    logistic_err = logistic(x_train, y_train, x_test, y_test)
    toc = perf_counter()
    print(f"Error for logistic regression: {logistic_err:.3f}")
    print(f"Time for logistic regression: {toc-tic:.3f} seconds")
    # MLP Classifier
    tic = perf_counter()
    mlp_err = mlp_class(x_train, y_train, x_test, y_test)
    toc = perf_counter()
    print(f"Error for MLP Classifier: {mlp_err:.3f}")
    print(f"Time for MLP : {toc-tic:.3f} seconds")
    # Deep Neural Network
    tic = perf_counter()
    nn_err = deep_nn(x_train, y_train, x_test, y_test)
    toc = perf_counter()
    print(f"Error for Deep NN: {nn_err:.3f}")
    print(f"Time for NN : {toc - tic:.3f} seconds")
