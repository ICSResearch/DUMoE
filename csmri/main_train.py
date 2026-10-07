import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_dataset
from moe_duns import load_sampling_matrix
from register_models import dumoe
from training.runner import get_args_parser, run

if __name__ == "__main__":
    args = get_args_parser("csmri", Path(__file__).resolve().parent).parse_args()
    run(args, "csmri", build_dataset, dumoe, load_sampling_matrix)
