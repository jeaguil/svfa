from pathlib import Path

STATIC_ROOT_PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_FOLDER = STATIC_ROOT_PARENT_PATH / "data"

ECMWF = DATA_FOLDER / "ecmwf_solar.csv"

ECMWF_COLUMNS = [
    "ZONEID",
    "TIMESTAMP",
    "VAR78",
    "VAR79",
    "VAR157",
    "VAR164",
    "VAR165",
    "VAR166",
    "VAR167",
    "VAR169",
    "VAR175",
    "VAR178",
    "VAR228",
    "POWER",
]

OUTFILE_TRAIN = DATA_FOLDER / "training_set.csv"
OUTFILE_TEST = DATA_FOLDER / "testing_set.csv"
