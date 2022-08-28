import sys
import argparse
import unittest

from pathlib import Path

from tests import test_scalers

test_choices = ["scalars"]


def cmp_preprocessing_scalars():
    test_scalers.cmp_scalars()


class TestSuite(unittest.TestCase, object):

    # option = ""

    # def __init__(self, *args, **kwargs):
    #     super(TestSuite, self).__init__(*args, **kwargs)
    #     self.args = sys.argv

    @unittest.skip("skip")
    def test_nothing(self):
        self.fail("Shouldn't happen")

    # @unittest.skipIf(sys.argv[1:] != "-s",
    #                  "Skipping testing of scalars")
    def test_cmp_preprocessing_scalars(self):
        cmp_preprocessing_scalars()

    def tearDown(self):
        return super().tearDown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalars", "-s", action="store_true")
    parser.add_argument("unittest_args", nargs="*")
    parser.set_defaults(scalars=False, models=False)

    args = parser.parse_args()

    sys.argv[1:] = args.unittest_args

    unittest.main()
