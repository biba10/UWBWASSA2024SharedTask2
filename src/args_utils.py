import argparse

from src.task import Task


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--model", type=str, default="google/mt5-large", help="Model to use")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    parser.add_argument("--accumulate_grad_batches", type=int, default=2,
                        help="Number of steps to accumulate before backprop")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--optimizer", type=str, default="adafactor", help="Optimizer")

    parser.add_argument("--scheduler", type=str, default="constant", help="Scheduler")

    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")

    parser.add_argument("--use_cpu", action="store_true", help="Use CPU")

    parser.add_argument("--max_train_data", type=int, default=None, help="Max train data")

    parser.add_argument("--max_dev_data", type=int, default=None, help="Max dev data")

    parser.add_argument("--max_test_data", type=int, default=None, help="Max test data")

    parser.add_argument("--inference_only", action="store_true", help="Inference only")

    parser.add_argument("--no_wandb", action="store_true", help="No wandb")

    parser.add_argument("--train_translated", action="store_true", default=False, help="Train on translated data")

    parser.add_argument("--train_acs", action="store_true", default=False, help="Train on ACS data")

    parser.add_argument("--load_test_data", action="store_true", help="Load test data")

    parser.add_argument("--id", type=str, default=None, help="ID of the run")

    args = parser.parse_args()
    return args


