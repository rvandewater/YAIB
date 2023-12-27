import torch
from typing import Callable
import numpy as np
from ignite.metrics import EpochMetric
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import jensenshannon
from torchmetrics.classification import BinaryFairness
from icu_benchmarks.models.similarity_func import correlation_spearman, distance_euclidean, correlation_pearson, cosine


""""
This file contains custom metrics that can be added to YAIB.
"""


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class BalancedAccuracy(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(BalancedAccuracy, self).__init__(
            self.balanced_accuracy_compute, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

        def balanced_accuracy_compute(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
            y_true = y_targets.numpy()
            y_pred = np.argmax(y_preds.numpy(), axis=-1)
            return balanced_accuracy_score(y_true, y_pred)


class CalibrationCurve(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(CalibrationCurve, self).__init__(
            self.ece_curve_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

        def ece_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, n_bins=10) -> float:
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return calibration_curve(y_true, y_pred, n_bins=n_bins)


class MAE(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        invert_transform: Callable = lambda x: x,
    ) -> None:
        super(MAE, self).__init__(
            lambda x, y: mae_with_invert_compute_fn(x, y, invert_transform),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )

        def mae_with_invert_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn=Callable) -> float:

            y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
            y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]

            return mean_absolute_error(y_true, y_pred)


class JSD(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
    ) -> None:
        super(JSD, self).__init__(
            lambda x, y: JSD_fn(x, y),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )

        def JSD_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
            return jensenshannon(abs(y_preds).flatten(), abs(y_targets).flatten()) ** 2


class TorchMetricsWrapper:
    metric = None

    def __init__(self, metric) -> None:
        self.metric = metric

    def update(self, output_tuple) -> None:
        self.metric.update(output_tuple[0], output_tuple[1])

    def compute(self) -> None:
        return self.metric.compute()

    def reset(self) -> None:
        return self.metric.reset()


class BinaryFairnessWrapper(BinaryFairness):
    """
    This class is a wrapper for the BinaryFairness metric from TorchMetrics.
    """

    group_name = None

    def __init__(self, group_name="sex", *args, **kwargs) -> None:
        self.group_name = group_name
        super().__init__(*args, **kwargs)

    def update(self, preds, target, data, feature_names) -> None:
        """ " Standard metric update function"""
        groups = data[:, :, feature_names.index(self.group_name)]
        group_per_id = groups[:, 0]
        return super().update(preds=preds.cpu(), target=target.cpu(), groups=group_per_id.long().cpu())

    def feature_helper(self, trainer, step_prefix):
        """Helper function to get the feature names from the trainer"""
        if step_prefix == "train":
            feature_names = trainer.train_dataloader.dataset.features
        elif step_prefix == "val":
            feature_names = trainer.train_dataloader.dataset.features
        else:
            feature_names = trainer.test_dataloaders.dataset.features
        return feature_names

# XAI Metrics


class Faithfulness(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(output_transform,
                         check_compute_fn, *args, **kwargs
                         )

    def update(
            self,
            x,
            attribution,
            model,
            similarity_func=None,
            nr_runs=100,
            pertrub=None,
            subset_size=3,
            feature=False,
            time_step=False,
            feature_timestep=False,
            device='cuda'

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
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.
        """
        def add_noise(x, indices, time_step, feature_timestep):
            noise = torch.randn_like(x["encoder_cont"])
            if time_step:
                idx0, idx1 = np.meshgrid(indices[0], indices[1], indexing='ij')

                with torch.no_grad():
                    x["encoder_cont"][idx0, idx1, :] += noise[idx0, idx1, :]

            elif feature:
                idx0, idx1 = np.meshgrid(indices[0], indices[1], indexing='ij')

                with torch.no_grad():
                    x["encoder_cont"][idx0, :, idx1] += noise[idx0, :, idx1]

            elif feature_timestep:
                idx0, idx1, idx2 = np.meshgrid(indices[0], indices[1], indices[2], indexing='ij')

                with torch.no_grad():
                    x["encoder_cont"][idx0, idx1, idx2] += noise[idx0, idx1, idx2]

        def apply_baseline(x, indices, time_step, feature_timestep):
            mask = torch.ones_like(x["encoder_cont"])
            if time_step:

                idx0, idx1, = np.meshgrid(indices[0], indices[1], indexing='ij')

                mask[idx0, idx1, :] -= mask[idx0, idx1, :]
            elif feature:
                idx0, idx1, = np.meshgrid(indices[0], indices[1], indexing='ij')

                mask[idx0, :, idx1] -= mask[idx0, :, idx1]

            elif feature_timestep:
                idx0, idx1, idx2 = np.meshgrid(indices[0], indices[1], indices[2], indexing='ij')

                mask[idx0, idx1, idx2] -= mask[idx0, idx1, idx2]

            with torch.no_grad():
                x["encoder_cont"] *= mask
        # Assuming 'attribution' is already a GPU tensor
        if not torch.is_tensor(attribution):
            attribution = torch.tensor(attribution).to(device)

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
                add_noise(x, a_ix, time_step, feature_timestep)
            elif pertrub == "baseline":
                apply_baseline(x, a_ix, time_step, feature_timestep)

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

                att_sums.append((attribution[patient_idx, :, :]
                                [:, timesteps_idx, :][:, :, feature_idx]).sum())

            pred_deltas.append((y_pred - y_pred_perturb)[patient_idx].item())
            # Convert to CPU for numpy operations

        pred_deltas_cpu = torch.tensor(pred_deltas).cpu().numpy()
        att_sums_cpu = torch.tensor(att_sums).cpu().numpy()

        similarities.append(similarity_func(pred_deltas_cpu, att_sums_cpu))

        score = np.nanmean(similarities)
        return score


class Stability(EpochMetric):
    def __init__(
        self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(output_transform,
                         check_compute_fn, *args, **kwargs
                         )

    def update(self, x,
               attribution, model, explain_method, method_name, dataloader=None, thershold=0.5, device='cuda', **kwargs
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
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.

        """

        def relative_stability_objective(
            x, xs, e_x, e_xs, close_indices, eps_min=0.0001, input=False, attention=False, device='cuda'
        ) -> torch.Tensor:
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
                    return torch.index_select(torch.tensor(input_array).to(device), 0, close_indices)

                return torch.index_select(input_array.to(device), 0, close_indices)

            # Convert all inputs to tensors and move to GPU
            if attention:
                x, xs = map(to_tensor, [x, xs])
            else:
                x, xs, e_x, e_xs = map(to_tensor, [x, xs, e_x, e_xs])

            if input:
                num_dim = x.ndim
            else:
                num_dim = e_x.ndim

            if num_dim == 3:
                def norm_function(arr): return torch.norm(arr, dim=(-1, -2))
            elif num_dim == 2:
                def norm_function(arr): return torch.norm(arr, dim=-1)
            else:
                def norm_function(arr): return torch.norm(arr)

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

        if explain_method == "Attention":
            y_pred = model.model.predict(dataloader)
            x_original = dataloader.dataset.data["reals"].clone()

            dataloader.dataset.add_noise()
            x_preturb = dataloader.dataset.data["reals"].clone()
            y_pred_preturb = model.model.predict(dataloader)
            Attention_weights = model.interpertations(dataloader)
            att_preturb = Attention_weights["attention"]
            # Calculate the absolute difference
            difference = torch.abs(y_pred_preturb - y_pred)

            # Find where the difference is less than or equal to a thershold
            close_indices = torch.nonzero(difference <= thershold).squeeze()[:, 0].to(device)

            RIS = relative_stability_objective(
                x_original.detach(), x_preturb.detach(), attribution, att_preturb, close_indices=close_indices, input=True, attention=True
            )
            ROS = relative_stability_objective(
                y_pred, y_pred_preturb, attribution, att_preturb, close_indices=close_indices, input=False, attention=True
            )

        else:
            y_pred = model(model.prep_data(x)).detach()
            x_original = x["encoder_cont"].detach().clone()

            with torch.no_grad():
                noise = torch.randn_like(x["encoder_cont"])*0.01
                x["encoder_cont"] += noise
            y_pred_preturb = model(model.prep_data(x)).detach()
            if explain_method == "Random":
                att_preturb = np.random.normal(size=[64, 24, 53])

            else:

                data, baselines = model.prep_data_captum(x)

                explantation = explain_method(model.forward_captum)
                # Reformat attributions.
                if explain_method is not captum.attr.Saliency:
                    att_preturb = explantation.attribute(data, baselines=baselines, **kwargs)
                else:
                    att_preturb = explantation.attribute(data, **kwargs)

                # Process and store the calculated attributions
                att_preturb = att_preturb[1].detach() if method_name in [
                    'Lime', 'FeatureAblation'] else torch.stack(att_preturb).detach()
            # Calculate the absolute difference
            difference = torch.abs(y_pred_preturb - y_pred)

            # Find where the difference is less than or equal to a thershold
            close_indices = torch.nonzero(difference <= thershold).squeeze()[:, 0].to(device)

            RIS = relative_stability_objective(
                x_original.detach(),
                x["encoder_cont"].detach(),
                attribution, att_preturb, close_indices=close_indices, input=True
            )
            ROS = relative_stability_objective(
                y_pred, y_pred_preturb, attribution, att_preturb, close_indices=close_indices, input=False
            )

        return np.max(RIS.cpu().numpy()).astype(np.float64), np.max(ROS.cpu().numpy()).astype(np.float64)


class Randomization(EpochMetric):
    def __init__(
        self, output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(output_transform,
                         check_compute_fn, *args, **kwargs
                         )

    def update(
        self, x,
        attribution, model, explain_method, random_model, similarity_func=cosine, dataloader=None, method_name="", **kwargs
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
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.

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
            y_pred = model(data).detach()

            explantation = explain_method(random_model.forward_captum)
            # Reformat attributions.
            if explain_method is not captum.attr.Saliency:
                attr = explantation.attribute(data, baselines=baselines, **kwargs)
            else:
                attr = explantation.attribute(data, **kwargs)

            # Process and store the calculated attributions
            random_attr = attr[1].cpu().detach().numpy() if method_name in [
                'Lime', 'FeatureAblation'] else torch.stack(attr).cpu().detach().numpy()

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
