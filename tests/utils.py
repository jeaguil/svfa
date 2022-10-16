# checks if necessary training files can be found for tests

from functools import wraps
import time
import logging
import pandas as pd

from pathlib import Path


class CustomAttributeError(Exception):
    """At attribute errors, consider running training in base directory."""


class TrainingFilesNotFounds(Exception):
    """Raise for training files not found in testing."""


X_path = Path(__file__).resolve().parent.parent / "data/X_training_df.csv"
X_full = pd.read_csv(X_path) if X_path.is_file() else None

Y_path = Path(__file__).resolve().parent.parent / "data/Y_training_df.csv"
Y_full = pd.read_csv(Y_path) if Y_path.is_file() else None


def setup_cli_logging():
    tests_log_file = Path(__file__).resolve().parent.parent / "logs/tests.log"

    if not Path(tests_log_file).is_file():
        p = Path(Path(__file__).resolve().parent.parent / "logs/")
        p.mkdir(parents=True, exist_ok=True)
        fn = "tests.log"
        filepath = p / fn

    logging.basicConfig(
        level=logging.INFO, handlers=[
            logging.FileHandler(
                tests_log_file),
            logging.StreamHandler()
        ])


def check_for_training_files():

    if X_full is None and Y_full is None:

        err_mes = " \n \
            Unable to correctly read in X and Y sets used for training.\n \
            Consider running ' python3 -m sfre --outTrainingSets ' in the base directory."

        raise TrainingFilesNotFounds(err_mes)

    else:
        return


def timer_fn(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        before = time.time()
        fn(*args, **kwargs)
        print("{} took: {} seconds".format(
            fn.__name__.upper(), time.time() - before))

    return wrapper
