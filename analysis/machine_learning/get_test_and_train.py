from sklearn.model_selection import train_test_split

"""
-----
reusing this several times so giving it its own function
-----
"""


def get_test_and_train(dataset):
    # getting test and train sets using sklearn built in

    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test
