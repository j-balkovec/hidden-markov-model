from src import hmm_main
from constants import *

import importlib.util
import os


def check_dependencies() -> bool:
    spec = importlib.util.find_spec("numpy")
    if spec is None:
        return False

    src_path = os.path.join(os.path.dirname(__file__), "src.py")
    constants_path = os.path.join(os.path.dirname(__file__), "constants.py")
    if not (os.path.exists(src_path) and os.path.exists(constants_path)):
        return False
    return True


def main() -> None:
    hmm_main(STATES_2_, OBSERVATIONS_2_, TRANSITION_PROB_2_,
             EMISSION_PROB_2_, INITIAL_PROB_2_, OBSERVATION_SEQ_2_)


if __name__ == "__main__":
    if not (check_dependencies()):
        print("Please make sure that numpy is installed and that src.py and constants.py are in the same directory as main.py.")
        exit()
    main()
