import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_dataset
from register_models import dumoe
from training.runner import get_args_parser, run

if __name__ == "__main__":
    args = get_args_parser("ics", Path(__file__).resolve().parent).parse_args()
    run(args, "ics", build_dataset, dumoe)
