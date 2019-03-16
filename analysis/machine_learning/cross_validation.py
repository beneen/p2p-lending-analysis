from sklearn.model_selection import GridSearchCV

"""
-----
cross validation
-----
"""


def cross_validation_best_parameters(dataset, model, param_grid):
    grid = GridSearchCV(model, param_grid,cv=10, scoring='accuracy')
    X=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    grid.fit(X,y)
    return grid.best_score_,grid.best_estimator_

