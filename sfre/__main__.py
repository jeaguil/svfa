import pandas as pd
import argparse
import sfre.const as sfre_consts

from sfre.split_data import SplitDataFile
from pathlib import Path


class DataFileError(BaseException):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solar Farm Data Analysis.")
    parser.add_argument("--split", "-s", action="store", default=None,
                        help="Split data based on optional arguments. (default: Split in set date range)",)
    args = parser.parse_args()
    args_dict = vars(args)

    if sfre_consts.ECMWF.is_file():
        ECMWF_Df = pd.read_csv(sfre_consts.ECMWF)
    else:
        raise DataFileError("Failed to initialize data file.")

    s = SplitDataFile(optional_args=args_dict["split"])

    if args_dict["split"] != None:
        """Split data in optional choices for --split flag.

        Options:
             """
        pass
    else:
        training_set, testing_set = pd.DataFrame(
            columns=sfre_consts.ECMWF_COLUMNS), pd.DataFrame(
            columns=sfre_consts.ECMWF_COLUMNS)

        if Path(sfre_consts.OUTFILE_TRAIN).is_file() and Path(sfre_consts.OUTFILE_TEST).is_file():
            training_set = pd.read_csv(sfre_consts.OUTFILE_TRAIN)
            testing_set = pd.read_csv(sfre_consts.OUTFILE_TEST)
        else:
            training_set, testing_set = s.split_data(ECMWF_Df, choice=None)
