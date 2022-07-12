import typing
import pandas as pd

from pathlib import Path
import sfre.const as sfre_consts


class SplitDataFile():
    def __init__(self, optional_args=None):
        self.optional_args = optional_args

    def split_data(self, _ECMWF, choice: typing.Optional[str]) -> pd.DataFrame:
        _ECMWF["TIMESTAMP"] = _ECMWF["TIMESTAMP"].apply(
            pd.to_datetime)
        if choice is None:
            """Default when there is not split in args.

            Split in interval date range. ECMWF Data ranges from 2012/04/01 to 2014/07/01.
            Default: 
                Training set: 2012/04/01 to 2013/07/01
                Testing set: 2013/07/01 to 2014/07/01

                Hour intervals.
            """
            training_set = _default_find_range(_ECMWF,
                                               "2012/04/01", "2013/07/01", sfre_consts.OUTFILE_TRAIN)

            testing_set = _default_find_range(
                _ECMWF, "2013/07/01", "2014/07/01", sfre_consts.OUTFILE_TEST)

            return training_set, testing_set


def _default_find_range(_data_file, start, end, out_file: str) -> pd.DataFrame:
    res_range_df = pd.DataFrame(columns=sfre_consts.ECMWF_COLUMNS)
    set_range = pd.date_range(start=start, end=end, freq="H")
    res_range_df = _data_file[_data_file["TIMESTAMP"].isin(
        set_range)]
    res_range_df.index = range(len(res_range_df))
    res_range_df.to_csv(out_file, index=False)
    return res_range_df
