import pandas as pd
import argparse
import logging
import sfre.const as sfre_consts

from sfre.split_data import SplitDataFile
from sfre.train import TrainModel
from pathlib import Path


class DataFileError(BaseException):
    pass


def start(args_dict):
    if sfre_consts.ECMWF.is_file():
        ECMWF_Df = pd.read_csv(sfre_consts.ECMWF)
    else:
        raise DataFileError("Failed to initialize data file.")

    training_set = pd.DataFrame(
        columns=sfre_consts.ECMWF_COLUMNS)
    testing_set = pd.DataFrame(
        columns=sfre_consts.ECMWF_COLUMNS)

    if args_dict["split"] != None:
        """Split data in optional choices for --split flag.

        Options:
             """
        logging.info(
            "Attempting to split data based on selected option from argument list.")
    else:
        if Path(sfre_consts.OUTFILE_TRAIN).is_file() and Path(sfre_consts.OUTFILE_TEST).is_file():
            logging.info(
                "Found local training and testing files. Collecting necessary information.")
            training_set = pd.read_csv(sfre_consts.OUTFILE_TRAIN)
            testing_set = pd.read_csv(sfre_consts.OUTFILE_TEST)
        else:
            logging.info(
                "Unable to find training or testing files. Attempting to construct with cli arguments.")
            s = SplitDataFile(optional_args=args_dict["split"])
            training_set, testing_set = s.split_data(ECMWF_Df, choice=None)

    if args_dict["train"]:
        logging.info(
            "Begin training model with found training set and selected model."
        )
        t = TrainModel(training_set, hours_ahead=24)


if __name__ == "__main__":

    main_log_file = sfre_consts.STATIC_ROOT_PARENT_PATH / "logs/main.log"
    if not Path(main_log_file).is_file():
        p = Path(sfre_consts.STATIC_ROOT_PARENT_PATH / "logs/")
        p.mkdir(parents=True, exist_ok=True)
        fn = "main.log"
        filepath = p / fn

    logging.basicConfig(
        level=logging.INFO, handlers=[
            logging.FileHandler(
                main_log_file),
            logging.StreamHandler()
        ])

    _model_choices = ["Default", "Temp"]

    parser = argparse.ArgumentParser(description="Solar Farm Data Analysis.")
    parser.add_argument("--split", "-s", action="store", default=None,
                        help="Split data based on optional arguments. (default: Split in set date range)")
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--no-train", "-nt", dest="train",
                        action="store_false")
    parser.set_defaults(train=False)
    args = parser.parse_args()
    args_dict = vars(args)

    start(args_dict)
