# from benchmark import Benchmark
from logit import Logit, train_logit, predict
import numpy as np
import pandas as pd
# from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_data(train_size: float=0.8) -> tuple:
    """
    fetches data, returns
    x_train, x_test, y_train, y_test
    :param train_size: size of train data set, default 0.8
    """

    df = pd.read_csv("~/Projects/HW_ML/data/data-classification/haberman.data",
                     header=0,
                     )
    df.columns = ["Age", "Operation year", "Positive nodes", "Survival"]

    # let's fill the column with next values:
    # 0 if patient died within 5 years
    # 1 if patient survived
    df["Survival"] = -(df["Survival"] - 2)

    X = df[["Age", "Operation year", "Positive nodes"]]
    x_train, x_test, y_train, y_test = train_test_split(X, df["Survival"], train_size=train_size)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy().reshape(y_train.shape[0],1)
    y_test = y_test.to_numpy().reshape(y_test.shape[0],1)

    return x_train, x_test, y_train, y_test


def logit_cls_experiment():
    x_train, x_test, y_train, y_test = get_data()

    alphas = [5, 1, 0.5, 0.1, 0.01, 0.001]
    train_accuracies = []
    test_accuracies = []

    for alpha in alphas:
        model = Logit(lr=alpha).fit(x_train, y_train)
        diff = model.predict(x_train) - y_train
        train_accuracies.append(
            (diff[diff == 0]+1).sum() / y_train.shape[0]
        )
        diff = model.predict(x_test) - y_test
        test_accuracies.append(
            (diff[diff == 0]+1).sum() / y_test.shape[0]
        )

    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    best_train, train_accuracy = alphas[np.argmax(train_accuracies)], np.max(train_accuracies)
    best_valid, test_accuracy = alphas[np.argmax(test_accuracies)], np.max(test_accuracies)

    print(f"Best training accuracy {train_accuracy * 100:.2f}% with lr={best_train}")
    print(f"Best validation accuracy {test_accuracy * 100:.2f}% with lr={best_valid}")


def logit_fun_experiment():
    x_train, x_test, y_train, y_test = get_data()

    alphas = [10, 5, 1, 0.5, 0.1, 0.01, 0.001]
    train_accuracies = []
    test_accuracies = []

    for alpha in alphas:
        model = train_logit(x_train, y_train, lr=alpha)
        diff = predict(x_train, model) - y_train
        train_accuracies.append(
            (diff[diff == 0]+1).sum() / y_train.shape[0]
        )
        diff = predict(x_test, model) - y_test
        test_accuracies.append(
            (diff[diff == 0]+1).sum() / y_test.shape[0]
        )

    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    best_train, train_accuracy = alphas[np.argmax(train_accuracies)], np.max(train_accuracies)
    best_valid, test_accuracy = alphas[np.argmax(test_accuracies)], np.max(test_accuracies)

    print(f"Best training accuracy {train_accuracy * 100:.2f}% with lr={best_train}")
    print(f"Best validation accuracy {test_accuracy * 100:.2f}% with lr={best_valid}")


if __name__ == "__main__":
    logit_cls_experiment()
    logit_fun_experiment()
