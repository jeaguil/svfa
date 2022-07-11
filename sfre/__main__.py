import pandas as pd
import argparse

from pathlib import Path

from sfre.split_data import SplitDataFile

STATIC_ROOT_PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_FOLDER = STATIC_ROOT_PARENT_PATH / "data"

ECMWF = DATA_FOLDER / "ecmwf_solar.csv"


class DataFileError(BaseException):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solar Farm Data Analysis.")
    parser.add_argument("--split", "-s", action="store", default=None,
                        help="Split data based on optional arguments. (default: Split in set date range)",)
    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict["split"] != None:
        s = SplitDataFile(optional_args=args_dict["split"])

    if ECMWF.is_file():
        ECMWF_Df = pd.read_csv(ECMWF)
    else:
        raise DataFileError("Failed to initialize data file.")
