from argparse import Namespace
import logging
import wandb


def wandb_running() -> bool:
    """Check if wandb is running."""
    return wandb.run is not None


def update_wandb_config(config: dict) -> None:
    """updates wandb config if wandb is running

    Args:
        config (dict): config to set
    """
    logging.info(f"updating config: {config}")
    if wandb_running():
        wandb.config.update(config)


def apply_wandb_sweep(args: Namespace) -> Namespace:
    """applies the wandb sweep configuration to the namespace object

    Args:
        args (Namespace): parsed arguments

    Returns:
        Namespace: arguments with sweep configuration applied (some are applied via hyperparams)
    """
    wandb.init()
    sweep_config = wandb.config
    args.__dict__.update(sweep_config)
    if args.hyperparams is None:
        args.hyperparams = []
    for key, value in sweep_config.items():
        args.hyperparams.append(f"{key}=" + (("'" + value + "'") if isinstance(value, str) else str(value)))
    logging.info(f"hyperparams after loading sweep config: {args.hyperparams}")
    return args


def wandb_log(log_dict):
    """logs metrics to wandb

    Args:
        log_dict (dict): metric dict to log
    """
    if wandb_running():
        wandb.log(log_dict)


def set_wandb_run_name(run_name):
    """stores the run name in wandb config

    Args:
        run_name (str): name of the current run
    """
    if wandb_running():
        wandb.config.update({"run-name": run_name})
        wandb.run.name = run_name
        wandb.run.save()
