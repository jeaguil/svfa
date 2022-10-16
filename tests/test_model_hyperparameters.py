""" 
Test model selection after data preprocessing.

Comparing the performance of tuned and untuned models."""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import pandas as pd

from utils import (check_for_training_files, X_full, Y_full,
                   CustomAttributeError, setup_cli_logging,
                   timer_fn)

from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

static_root_parent_path = Path(__file__).resolve().parent.parent
training_data_csv = static_root_parent_path / "data/training_set.csv"
training_df = pd.read_csv(training_data_csv)
feature_names = list(training_df.columns)


@timer_fn
def _fit_and_search(parameters, model_fn):
    clf = GridSearchCV(model_fn(), parameters)
    clf.fit(X_full, Y_full.values.ravel())

    _to_print = """
    {0}
    Best hyperparameters: {1}
    Found best score: {2}
    """.format(model_fn.__name__, clf.best_params_, clf.best_score_)
    print(_to_print)


if __name__ == "__main__":

    setup_cli_logging()

    # check if training sets are available
    check_for_training_files()

    try:
        X_full = X_full.drop(["ZONEID", "TIMESTAMP"], axis=1)
        Y_full = Y_full.drop(["Unnamed: 0"], axis=1)
    except AttributeError:
        """ NoneType object has no attribute 'drop' """
        raise CustomAttributeError("\n \
            Consider running ' python3 -m sfre --outTrainingSets ' in the base directory.")

    model_selections_parameters = [
        (RandomForestRegressor, {
        }),
        (Lasso, {
        }),
        (SGDRegressor, {
        }),
        (ElasticNet, {

        }),
        (SVR, {
        }),
        (KNeighborsRegressor, {
        }),
        (GradientBoostingRegressor, {
        }),
    ]

    for i in model_selections_parameters:
        _fit_and_search(parameters=i[1], model_fn=i[0])
