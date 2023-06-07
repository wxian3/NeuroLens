import ray
import time
import logging
import coloredlogs
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import click
from hydra.experimental import initialize, compose
from .optimize_marker import cli as opcli


LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--smoke",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="Finish quickly for testing.",
)
@click.option(
    "--max_epochs",
    type=click.INT,
    default=10000,
    help="Maximum number of steps per experiment.",
)
def cli(smoke=False, max_epochs=10000):
    """
    Optimize hyperparameters for pattern and detector optimization.

    Run as `pdm run hyperopt`.
    """
    LOGGER.info("Running hyperparameter search.")
    ray.init(num_gpus=1)
    LOGGER.info("Creating HyperBand scheduler...")
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=max_epochs,
        grace_period=1000,
    )
    # 'training_iteration' is incremented every time `trainable.step` is called
    stopping_criteria = {"training_iteration": 10 if smoke else max_epochs}
    LOGGER.info("Stopping criteria: %s.", stopping_criteria)
    LOGGER.info("Starting search...")
    analysis = tune.run(
        optimize_pattern_objective,
        name="optimize_pattern_asha",
        scheduler=scheduler,
        metric="accuracy",
        mode="min",
        stop=stopping_criteria,
        num_samples=20,
        verbose=1,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config={  # Hyperparameter space
            "model.lr": tune.loguniform(1e-4, 1e-1),
            "model.lr_fcn_fac": tune.loguniform(1e-2, 1e2),
            "model.lr_marker_fac": tune.loguniform(1e0, 1e4),
            "model.n_latent": tune.randint(20, 400),
            "model.n_hidden": tune.randint(1, 3),
        },
    )
    LOGGER.info("Best hyperparameters found were: ", analysis.best_config)


def optimize_pattern_objective(config):
    overrides = [f"{key}={val}" for key, val in config.items()]
    overrides.append(f"exp_name=hyperopt_{time.time()}")
    overrides.append(f"trainer.max_steps=10000")
    with initialize(config_path="../../conf"):
        cfg = compose(config_name="calibration_config", overrides=overrides)
    opcli(cfg)


if __name__ == "__main__":
    coloredlogs.install(level=logging.INFO)
    cli()
