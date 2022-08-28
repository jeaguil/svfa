#!/usr/bin/env python

import pandas as pd
import argparse
import logging
import sfre.const as sfre_consts
import sfre.train as sfret

from sfre.split_data import SplitDataFile
from sfre.train import TrainModelParams
from pathlib import Path


class DataFileError(BaseException):
    pass


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

    _split_choices = ["d"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", action="store", default=None,
                        help=": Split data based on optional arguments. (default: Split in set date range)",
                        choices=_split_choices)
    parser.add_argument("-t", "--train", action="store_true",
                        help=": Train machine learning model.")
    parser.add_argument("-st", "--skipTraining", dest="train",
                        action="store_false", help=": Skip training the machine learning model.")
    parser.add_argument("-ot", "--outTrainingSets", action="store_true",
                        help=": Output training X and training Y set dataframes into CSV files.")
    parser.set_defaults(train=False, outTrainingSets=False)
    args = parser.parse_args()
    args_dict = vars(args)

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
                "Found local training and testing CSV files. Collecting necessary information...")
            training_set = pd.read_csv(sfre_consts.OUTFILE_TRAIN)
            testing_set = pd.read_csv(sfre_consts.OUTFILE_TEST)
        else:
            logging.info(
                "Unable to find training or testing files. Attempting to construct with cli arguments.")
            s = SplitDataFile(optional_args=args_dict["split"])
            training_set, testing_set = s.split_data(ECMWF_Df, choice=None)

    training_model_parameters = None
    if args_dict["outTrainingSets"]:

        logging.info(
            "Output X and Y training sets to CSV files to data folder."
        )

        training_model_parameters = TrainModelParams(
            training_set, hours_ahead=24)

        if not Path(sfre_consts.OUTFILE_TRAIN_X).is_file():
            training_model_parameters.output_training_X_df()
        if not Path(sfre_consts.OUTFILE_TRAIN_Y).is_file():
            training_model_parameters.output_training_Y_df()

    if args_dict["train"]:
        logging.info(
            "Begin training model with found training set and selected model."
        )
        if training_model_parameters is None:
            training_model_parameters = TrainModelParams(
                training_set, hours_ahead=24)
