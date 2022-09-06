##Toxic Comments Classification from Kaggle

Data source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

###This is a multi label (Multiple label/output/target) with 2 possible classes (0 or 1) for each label, hence it's a multi-label binary classification.

##Future improvement - Implement a Gridsearch for this.

Grid search for this *multilabel AND binary classification* not yet implemented due to error I am not sure of.
GridSearchCV for multilabel (6 labels) + 2 classes FOR EACH label = multilabel and binary. since it's a binary classification with multiple output/lables/target, OVR can be deployed.

ValueError: Invalid parameter naive_classifier for estimator Pipeline(steps=[('multinomialnb', MultinomialNB())]). Check the list of available parameters with `estimator.get_params().keys()`.

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

##first attempt
params = {'alpha':[0,0.001,0.1,1,10,100]}
searcher = GridSearchCV(estimator=naive_model,param_grid=params)
searcher.fit(X_train,y_train)
searcher.best_params_

##second attempt
     naive_classifier = OneVsRestClassifier(
     make_pipeline(naive_model)
     Should achieve the same thing, except using Pipeline, and not make_pipeline, the step is explicitly specified
     Pipeline([("multinomialnb", naive_model)])
)

param_grid = {'estimator__naive_classifier__alpha': [0,0.001,0.1,1,10,100]}

grid_search = GridSearchCV(naive_classifier, param_grid=param_grid,cv=5, scoring = 'f1_micro')
grid_search.fit(X_train, y_train)
