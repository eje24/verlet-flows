# from https://github.com/gcorso/DiffDock/blob/main/utils/parsing.py
import logging
from argparse import Namespace, ArgumentParser, FileType

logging.basicConfig(level=logging.INFO)


def display_args(args: Namespace):
    logging.info("Starting training run with the following arguments:")
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        logging.info(f"{arg_name}: {arg_value}")


def parse_args(manual_args=None):
    # General arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=FileType(mode="r"), default=None)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="workdir",
        help="Folder in which to save model and logs",
    )
    parser.add_argument(
        "--restart_dir",
        type=str,
        help="Folder of previous training model from which to restart",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="")
    parser.add_argument("--project", type=str, default="verlet-flows", help="")
    parser.add_argument("--run_name", type=str, default="", help="")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=False,
        help="CUDA optimization parameter for faster training",
    )

    # Training arguments
    parser.add_argument(
        "--n_epochs", type=int, default=75, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--scheduler", type=str, default=None, help="LR scheduler")
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=20,
        help="Patience of the LR scheduler",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--restart_lr",
        type=float,
        default=None,
        help="If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.",
    )
    parser.add_argument(
        "--w_decay", type=float, default=0.0, help="Weight decay added to loss"
    )

    parser.add_argument(
        "--num_train", type=int, default=450, help="Size of training set"
    )
    parser.add_argument(
        "--num_val", type=int, default=50, help="Size of validation set"
    )
    parser.add_argument(
        "--num_integrator_steps", type=int, default=15, help="Number of integrator steps"
    )
    # GMM argument
    parser.add_argument(
        "--nmodes", type=int, default=3, help="Number of modes in the GMM"
    )

    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args
