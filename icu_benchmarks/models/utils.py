import json
from typing import Dict
from pathlib import Path
from datetime import timedelta
from enum import Enum
from json import JSONEncoder
import gin
import logging
import numpy as np
import torch
from quantus.functions.similarity_func import correlation_spearman, cosine
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import Module
from torch.optim import Optimizer, Adam, SGD, RAdam
from typing import Optional, Union
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR, ExponentialLR
import captum


def save_config_file(log_dir):
    config_path = log_dir / "train_config.gin"
    with config_path.open("w") as f:
        f.write(gin.operative_config_str())


def create_optimizer(name: str, model: Module, lr: float, momentum: float = 0) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    name = name.lower()
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
    scheduler_name: Optional[str],
    optimizer: Optimizer,
    lr_factor: float,
    lr_steps: Optional[list],
    epochs: int,
) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")


class JsonResultLoggingEncoder(JSONEncoder):
    """JSON converter for objects that are not serializable by default."""

    # Serializes foreign datatypes
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, tuple):
            if isinstance(obj)[0] is torch.Tensor or isinstance(obj)[0] is np.ndarray:
                return map(lambda item: item.tolist(), obj)
        if isinstance(obj, timedelta):
            return str(obj)
        return JSONEncoder.default(self, obj)


class Align(str, Enum):
    LEFT = "<"
    CENTER = "^"
    RIGHT = ">"


def log_table_row(
    cells: list,
    level: int = logging.INFO,
    widths: list[int] = None,
    header: list[str] = None,
    align: Align = Align.LEFT,
    highlight: bool = False,
):
    """Logs a table row.

    Args:
        cells: List of cells to log.
        level: Logging level.
        widths: List of widths for each cell.
        header: List of headers to calculate widths if widths not supplied.
        highlight: If set to true, highlight the row.
    """
    table_cells = cells
    if not widths and header:
        widths = [len(head) for head in header]
    if widths:
        table_cells = []
        for cell, width in zip(cells, widths):
            cell = str(cell)[:width]  # truncate cell if it is too long
            table_cells.append("{: {align}{width}}".format(cell, align=align.value, width=width))
    table_row = " | ".join([f"{cell}" for cell in table_cells])
    if highlight:
        table_row = f"\x1b[31;32m{table_row}\x1b[0m"
    logging.log(level, table_row)


class JSONMetricsLogger(Logger):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        if output_dir is None:
            output_dir = Path.cwd() / "metrics"
        logging.info(f"logging metrics to file: {str(output_dir.resolve())}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return "json_metrics_logger"

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        old_metrics = {}
        stage_metrics = {
            "train": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("train/")},
            "val": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("val/")},
            "test": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("test/")},
        }
        for stage, metrics in stage_metrics.items():
            if metrics:
                output_file = self.output_dir / f"{stage}_metrics.json"
                old_metrics = {}
                if output_file.exists():
                    try:
                        with output_file.open("r") as f:
                            old_metrics = json.load(f)
                        logging.debug(f"updating {stage} metrics file...")
                    except json.decoder.JSONDecodeError:
                        logging.warning("could not decode json file, overwriting...")

                old_metrics.update(metrics)
                with output_file.open("w") as f:
                    json.dump(old_metrics, f, cls=JsonResultLoggingEncoder, indent=4)

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass


def Faithfulness_Correlation(
    model,
    x,
    attribution,
    similarity_func=None,
    nr_runs=100,
    pertrub=None,
    subset_size=3,
    feature=False,
    time_step=False,
    feature_timestep=False,
):
    """
    Calculates faithfulness scores for captum attributions

    Args:
        - x:Batch input
        -attribution: attribution generated by captum,
        - similarity_func:function to determine similarity between sum of attributions and difference in prediction
        - nr_runs: How many times to repeat the experiment,
        - pertrub: What change to do to the input,
        - subset_size: The size of the subset of featrues to alter ,
        - feature: Determines if to calcualte faithfulness of feature attributions,
        - time_step: Determines if to calcualte faithfulness of timesteps attributions,
        - feature_timestep: Determines if to calcualte faithfulness of featrues per timesteps attributions,
    Returns:
        score: similarity score between sum of attributions and difference in prediction averaged over nr_runs

    Implementation of faithfulness correlation by Bhatt et al., 2020.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.
    This code is adapted from the quantus libray to suit our use case

    References:
        1) Umang Bhatt et al.: "Evaluating and aggregating feature-based model
        explanations." IJCAI (2020): 3016-3022.
        2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for
        responsible evaluation of neural network explanations and beyond."
        Journal of Machine Learning Research 24.34 (2023): 1-11.
    """

    attribution = torch.tensor(attribution).to(model.device)

    # Other initializations
    if similarity_func is None:
        similarity_func = correlation_spearman
    if pertrub is None:
        pertrub = "baseline"
    similarities = []

    # Assuming this is a method to prepare your data

    y_pred = model(model.prep_data(x)).detach()  # Keep on GPU
    pred_deltas = []
    att_sums = []

    for i_ix in range(nr_runs):
        if time_step:
            timesteps_idx = np.random.choice(24, subset_size, replace=False)
            patient_idx = np.random.choice(64, 1, replace=False)
            a_ix = [patient_idx, timesteps_idx]

        elif feature:
            feature_idx = np.random.choice(53, subset_size, replace=False)
            patient_idx = np.random.choice(64, 1, replace=False)
            a_ix = [patient_idx, feature_idx]
        elif feature_timestep:
            timesteps_idx = np.random.choice(24, subset_size[0], replace=False)
            feature_idx = np.random.choice(53, subset_size[1], replace=False)
            patient_idx = np.random.choice(64, 1, replace=False)
            a_ix = [patient_idx, timesteps_idx, feature_idx]

        # Apply perturbation
        if pertrub == "Noise":
            x = model.add_noise(x, a_ix, time_step, feature, feature_timestep)
        elif pertrub == "baseline":
            x = model.apply_baseline(x, a_ix, time_step, feature, feature_timestep)

        # Predict on perturbed input and calculate deltas
        y_pred_perturb = (model(model.prep_data(x))).detach()  # Keep on GPU

        if time_step:
            if attribution.size() == torch.Size([24]):
                att_sums.append((attribution[timesteps_idx]).sum())
            else:
                att_sums.append((attribution[patient_idx, :, :][:, timesteps_idx, :]).sum())
        elif feature:
            if len(attribution) == 53:
                att_sums.append((attribution[feature_idx]).sum())
            else:
                att_sums.append((attribution[patient_idx, :, :][:, :, feature_idx]).sum())
        elif feature_timestep:
            att_sums.append((attribution[patient_idx, :, :][:, timesteps_idx, :][:, :, feature_idx]).sum())

        pred_deltas.append((y_pred - y_pred_perturb)[patient_idx].item())
        # Convert to CPU for numpy operations

    pred_deltas_cpu = torch.tensor(pred_deltas).cpu().numpy()
    att_sums_cpu = torch.tensor(att_sums).cpu().numpy()

    similarities.append(similarity_func(pred_deltas_cpu, att_sums_cpu))

    score = np.nanmean(similarities)
    return score


def Data_Randomization(
    model,
    x,
    attribution,
    explain_method,
    random_model,
    similarity_func=cosine,
    dataloader=None,
    method_name="",
    **kwargs,
):
    """

    Args:
        - x:Batch input
        -attribution: attribution
        - explain_method:function to generate explantations
        - random_model: Reference to model trained on random labels
        - similarity_func: Function to measure similiarity
        - dataloader:In case of using Attention as the explain method need to pass the dataloader instead of the batch ,
        - method_name: Name of the explantation

    Returns:
        score: similarity score between attributions of model trained on random data and model trained on real data

    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.
    This code is adapted from the quantus libray to suit our use case

    References:
        1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
        Attributions Fail." ICML (2020): 9046-9057.
        2)Hedström, Anna, et al. "Quantus: An explainable ai
        toolkit for responsible evaluation of neural network explanations and beyond."
            Journal of Machine Learning Research 24.34 (2023): 1-11.

    """

    if explain_method == "Attention":
        Attention_weights = random_model.interpertations(dataloader)
        attribution = attribution.cpu().numpy()
        min_val = np.min(attribution)
        max_val = np.max(attribution)

        attribution = (attribution - min_val) / (max_val - min_val)
        random_attr = Attention_weights["attention"].cpu().numpy()
        min_val = np.min(random_attr)
        max_val = np.max(random_attr)
        random_attr = (random_attr - min_val) / (max_val - min_val)
        score = similarity_func(random_attr, attribution)
    elif explain_method == "Random":
        score = similarity_func(np.random.normal(size=[64, 24, 53]).flatten(), attribution.flatten())
    else:
        data, baselines = model.prep_data_captum(x)

        explantation = explain_method(random_model.forward_captum)
        # Reformat attributions.
        if explain_method is not captum.attr.Saliency:
            attr = explantation.attribute(data, baselines=baselines, **kwargs)
        else:
            attr = explantation.attribute(data, **kwargs)

        # Process and store the calculated attributions
        random_attr = (
            attr[1].cpu().detach().numpy()
            if method_name in ["Lime", "FeatureAblation"]
            else torch.stack(attr).cpu().detach().numpy()
        )

        attribution = attribution.flatten()
        min_val = np.min(attribution)
        max_val = np.max(attribution)
        attribution = (attribution - min_val) / (max_val - min_val)
        random_attr = random_attr.flatten()
        min_val = np.min(random_attr)
        max_val = np.max(random_attr)
        random_attr = (random_attr - min_val) / (max_val - min_val)

        score = similarity_func(random_attr, attribution)
    return score


def Relative_Stability(
    model,
    x,
    attribution,
    explain_method,
    method_name,
    dataloader=None,
    threshold=0.5,
    **kwargs,
):
    """
    Args:
            - x:Batch input
            -attribution: attribution
            - explain_method:function to generate explantations
            - method_name: Name of the explantation
            - dataloader:In case of using Attention as the explain method need to pass the dataloader instead of the batch ,


        Returns:
            RIS : relative distance between the explantation and the input
            ROS: relative distance between the explantation and the output


    References:
            1) `https://arxiv.org/pdf/2203.06877.pdf
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation
            of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.

    """

    def relative_stability_objective(x, xs, e_x, e_xs, eps_min=0.0001, input=False, device="cuda") -> torch.Tensor:
        """
        Computes relative input and output stabilities maximization objective
        as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.

        Args:

            x: Input tensor
            xs: perturbed tensor.
            e_x: Explanations for x.
            e_xs: Explanations for xs.
            eps_min:Value to avoid division by zero if needed
            input:Boolean to indicate if this is an input or an output
            device: the device to keep the tensors on

        Returns:

            ris_obj: Tensor
                RIS maximization objective.
        """

        # Function to convert inputs to tensors if they are numpy arrays
        def to_tensor(input_array):
            if isinstance(input_array, np.ndarray):
                return torch.tensor(input_array).to(device)
            return input_array.to(device)

        # Convert all inputs to tensors and move to GPU
        x, xs, e_x, e_xs = map(to_tensor, [x, xs, e_x, e_xs])

        if input:
            num_dim = x.ndim
        else:
            num_dim = e_x.ndim

        if num_dim == 3:

            def norm_function(arr):
                return torch.norm(arr, dim=(-1, -2))

        elif num_dim == 2:

            def norm_function(arr):
                return torch.norm(arr, dim=-1)

        else:

            def norm_function(arr):
                return torch.norm(arr)

        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * eps_min)
        nominator = norm_function(nominator)

        if input:
            denominator = x - xs
            denominator /= x + (x == 0) * eps_min
            denominator = norm_function(denominator)
            denominator += (denominator == 0) * eps_min
        else:
            denominator = torch.squeeze(x) - torch.squeeze(xs)
            denominator = torch.norm(denominator, dim=-1)
            denominator += (denominator == 0) * eps_min

        return nominator / denominator

    attribution = torch.tensor(attribution).to(model.device)
    if explain_method == "Attention":
        y_pred = model.model.predict(dataloader)
        x_original = dataloader.dataset.data["reals"].clone()

        dataloader.dataset.add_noise()
        x_perturb = dataloader.dataset.data["reals"].clone()
        y_pred_perturb = model.model.predict(dataloader)
        Attention_weights = model.interpertations(dataloader)
        att_perturb = Attention_weights["attention"]
        # Calculate the absolute difference
        difference = torch.abs(y_pred_perturb - y_pred)

        # Find where the difference is less than or equal to a threshold
        close_indices = torch.nonzero(difference <= threshold).squeeze()
        RIS = relative_stability_objective(
            x_original[close_indices, :, :].detach(),
            x_perturb[close_indices, :, :].detach(),
            attribution,
            att_perturb,
            input=True,
        )

        ROS = relative_stability_objective(
            y_pred[close_indices],
            y_pred_perturb[close_indices],
            attribution,
            att_perturb,
            input=False,
        )

    else:
        y_pred = model(model.prep_data(x)).detach()
        x_original = x["encoder_cont"].detach().clone()

        with torch.no_grad():
            noise = torch.randn_like(x["encoder_cont"]) * 0.01
            x["encoder_cont"] += noise
        y_pred_perturb = model(model.prep_data(x)).detach()
        if explain_method == "Random":
            att_perturb = np.random.normal(size=[64, 24, 53])
            att_perturb = torch.tensor(att_perturb).to(model.device)
        else:
            data, baselines = model.prep_data_captum(x)

            explantation = explain_method(model.forward_captum)
            # Reformat attributions.
            if explain_method is not captum.attr.Saliency:
                att_perturb = explantation.attribute(data, baselines=baselines, **kwargs)
            else:
                att_perturb = explantation.attribute(data, **kwargs)

            # Process and store the calculated attributions
            att_perturb = (
                att_perturb[1].detach() if method_name in ["Lime", "FeatureAblation"] else torch.stack(att_perturb).detach()
            )
        # Calculate the absolute difference
        difference = torch.abs(y_pred_perturb - y_pred)

        # Find where the difference is less than or equal to a threshold
        close_indices = torch.nonzero(difference <= threshold).squeeze()
        RIS = relative_stability_objective(
            x_original[close_indices, :, :].detach(),
            x["encoder_cont"][close_indices, :, :].detach(),
            attribution[close_indices, :, :],
            att_perturb[close_indices, :, :],
            input=True,
        )
        ROS = relative_stability_objective(
            y_pred[close_indices],
            y_pred_perturb[close_indices],
            attribution[close_indices, :, :],
            att_perturb[close_indices, :, :],
            input=False,
        )

    return np.max(RIS.cpu().numpy()).astype(np.float64), np.max(ROS.cpu().numpy()).astype(np.float64)
