# from https://github.com/gcorso/DiffDock/blob/main/utils/parsing.py
import logging
from argparse import Namespace, ArgumentParser, FileType

logging.basicConfig(level=logging.INFO)


def display_args(args: Namespace):
    logging.info("Starting training run with the following arguments:")
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        logging.info(f"{arg_name}: {arg_value}")


def parse_train_args(manual_args=None):
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
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/cache",
        help="Folder from where to load/restore cached dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PDBBind_processed/",
        help="Folder containing original structures",
    )
    parser.add_argument(
        "--split_train",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_train",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_val",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_val",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        default="data/splits/timesplit_test",
        help="Path of file defining the split",
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
    parser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader",
    )

    # Training arguments
    parser.add_argument(
        "--n_epochs", type=int, default=400, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--overfit",
        action="store_true",
        default=False,
        help="Whether or not to overfit to a single structure/pose",
    )
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
        "--num_workers", type=int, default=1, help="Number of workers for preprocessing"
    )

    # Dataset
    parser.add_argument(
        "--num_train", type=int, default=450, help="Size of training set"
    )
    parser.add_argument(
        "--num_val", type=int, default=50, help="Size of validation set"
    )

    # Model
    parser.add_argument(
        "--num_coupling_layers",
        type=int,
        default=5,
        help="Number of coupling layers in the flow",
    )
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of interaction layers"
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=16,
        help="Number of hidden features per node of order 0",
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=4,
        help="Number of hidden features per node of order >0",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument(
        "--distance_embed_dim",
        type=int,
        default=6,
        help="Dimension of distance embedding.",
    )

    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args
