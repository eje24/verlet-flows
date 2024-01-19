# from https://github.com/gcorso/DiffDock/blob/main/utils/parsing.py
import logging
import argparse
from argparse import Namespace, ArgumentParser, FileType

logging.basicConfig(level=logging.INFO)


def display_args(args: Namespace):
    logging.info("Starting training run with the following arguments:")
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        logging.info(f"{arg_name}: {arg_value}")

def parse_cnf_args(manual_args = None):
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--run_name', type=str, default='2d_cnf')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_train', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=256)
    # Flow
    parser.add_argument('--residual', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_hidden_units', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=25)
    # Source
    parser.add_argument('--source', type=str, default='gaussian')
    parser.add_argument('--source_nmode', type=int, default=2)
    # Target
    parser.add_argument('--target', type=str, default='gmm')
    parser.add_argument('--target_nmode', type=int, default=3)
    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args

def parse_phase_space_cnf_args(manual_args = None):
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--run_name', type=str, default='verlet_cnf')
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_train', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=256)
    # Flow Type
    parser.add_argument(
        "--verlet", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use VerletFlow and VerletIntegrator"
    )
    # Non-Verlet Flow
    parser.add_argument('--num_hidden_units', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    # Verlet Flow
    parser.add_argument('--num_vp_hidden', type=int, default=16)
    parser.add_argument('--num_vp_layers', type=int, default=3)
    parser.add_argument('--num_nvp_hidden', type=int, default=16)
    parser.add_argument('--num_nvp_layers', type=int, default=3)
    # Integrator
    parser.add_argument('--num_timesteps', type=int, default=25)
    # Source
    parser.add_argument('--source', type=str, default='gaussian')
    parser.add_argument('--source_nmode', type=int, default=2)
    # Target
    parser.add_argument('--target', type=str, default='gmm')
    parser.add_argument('--target_nmode', type=int, default=2)
    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args

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
        "--n_epochs", type=int, default=25, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--scheduler", type=str, default="plateau", help="LR scheduler")
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
        "--num_train", type=int, default=4096, help="Size of training set"
    )
    parser.add_argument(
        "--num_val", type=int, default=1024, help="Size of validation set"
    )
    parser.add_argument(
        "--num_integrator_steps", type=int, default=8, help="Number of integrator steps"
    )
    parser.add_argument(
        "--loss", type=str, default="likelihood_loss", help="Loss function to use"
    )
    # Flow arguments
    parser.add_argument(
        "--verlet", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use VerletFlow and VerletIntegrator"
    )
    parser.add_argument(
        "--source", type=str, default="gaussian", help="Type of flow to use"
    )
    parser.add_argument(
        "--target", type=str, default="gmm", help="Target distribution"
    )
    parser.add_argument(
        "--data_dim", type=int, default=2, help="Dimension of the q distributions"
    )
    # VerletFlow-specific arguments
    parser.add_argument(
        "--num_vp_hidden_layers", type=int, default=5, help="Number of hidden layers"
    )
    parser.add_argument(
        "--num_vp_hidden_units", type=int, default=50, help="Number of hidden units"
    )
    parser.add_argument(
        "--num_nvp_hidden_layers", type=int, default=5, help="Number of hidden layers"
    )
    parser.add_argument(
        "--num_nvp_hidden_units", type=int, default=50, help="Number of hidden units"
    )
    # Source Gaussian arguments
    parser.add_argument(
        "--source_gaussian_mean", type=float, default=0.0, help="Mean of the source Gaussian"
    )
    parser.add_argument(
        "--source_gaussian_xvar", type=float, default=1.0, help="x variance of the source Gaussian"
    )
    parser.add_argument(
        "--source_gaussian_yvar", type=float, default=1.0, help="y variance of the source Gaussian"
    )
    parser.add_argument(
        "--source_gaussian_xyvar", type=float, default=0.0, help="xy covariance of the source Gaussian"
    )
    # Source GMM argument
    parser.add_argument(
        "--source_nmodes", type=int, default=2, help="Number of modes in the source GMM"
    )

    # Target Gaussian arguments
    parser.add_argument(
        "--target_gaussian_mean", type=float, default=1.0, help="Mean of the target Gaussian"
    )
    parser.add_argument(
        "--target_gaussian_xvar", type=float, default=4.0, help="x variance of the target Gaussian"
    )
    parser.add_argument(
        "--target_gaussian_yvar", type=float, default=1.0, help="y variance of the target Gaussian"
    )
    parser.add_argument(
        "--target_gaussian_xyvar", type=float, default=1.0, help="xy covariance of the target Gaussian"
    )

    # Target GMM argument
    parser.add_argument(
        "--target_nmodes", type=int, default=3, help="Number of modes in the target GMM"
    )


    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args
