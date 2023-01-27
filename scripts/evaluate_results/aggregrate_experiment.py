import json
import math
from pathlib import Path
import pandas as pd


def aggregate_results(
    log_dir: Path,
    models=["LSTM", "Transformer", "GRU", "RNN", "LogisticRegression", "LGBMClassifier", "TCN"],
    metric_type="AUC",
    include_unfinished=False,
    iterations=5,
    decimals=2,
    scale=100,
    sort=["Dataset", "Model"],
    datasets=["miiv", "aumc", "hirid", "eicu"],
    results_file="accumulated_test_metrics.json"
):
    """
    Aggregate results from a log directory.
    Args:
        log_dir: Log directory stub.
        models: List of models to include in the results.
        metric_type: Metric to aggregate.
        include_unfinished: Include experiments that have not finished.
        iterations: Correct standard deviation for amount of iterations.
        decimals: Calculate results to x decimals.
        sort: Where to sort on.
        datasets: Which datasets to include.
        results_file: Name of the results file.
    """
    results = pd.DataFrame(columns=["Time", "Dataset", "Model", "Average", "Std", "95% CI", "Execution Time"])
    for dataset in log_dir.iterdir():
        if dataset.is_dir() and dataset.name in datasets:
            for task in dataset.iterdir():
                for model in task.iterdir():
                    for log_time in model.iterdir():
                        if Path(log_time / results_file).exists():
                            test_metrics = json.load(open(log_time / results_file))
                            avg = test_metrics["avg"][metric_type]
                            std = test_metrics["std"][metric_type]
                            ci_95 = test_metrics["CI_0.95"][metric_type]
                            time = "NaN"
                            if "execution_time" in test_metrics:
                                time = test_metrics["execution_time"]
                            results.loc[len(results.index)] = [log_time.name, dataset.name, model.name, avg, std, ci_95, time]

                        elif include_unfinished:
                            avg = "NaN"
                            std = "NaN"
                            ci_95 = "NaN"
                            time = "NaN"
                            results.loc[len(results.index)] = [log_time.name, dataset.name, model.name, avg, std, ci_95, time]

    # Exclude nan rows for calculations
    nan_rows = results[results["95% CI"] == "NaN"]
    results = results[results["95% CI"] != "NaN"]

    # Convert to units
    results["Execution Time"] = pd.to_timedelta(results["Execution Time"])
    results["Average"] = pd.to_numeric(results["Average"], errors="coerce")
    results["Std"] = pd.to_numeric(results["Std"], errors="coerce")

    # Scaling
    results["95% CI"] = results["95% CI"].apply(lambda x: (pd.to_numeric(x[1]) * scale, pd.to_numeric(x[0]) * scale))
    results[["Average", "Std"]] = results[["Average", "Std"]].apply(lambda x: x * scale)

    # Correct std for amount of iterations
    if iterations > 1:
        results["Std"] = results["Std"].apply(lambda x: x / math.sqrt(iterations))

    # Round everything to x decimals
    results = results.round(decimals)
    results["95% CI"] = results["95% CI"].apply(lambda x: tuple(map(lambda y: round(y, decimals), x)))

    # Append unfinished results
    results = pd.concat([results, nan_rows])

    # Sort
    results = results.set_index(["Dataset"])
    results = results.sort_values(by=sort, ascending=[True, True])

    # Exclude datasets
    results = results[results["Model"].isin(models)]
    print(results.to_markdown())


aggregate_results(Path(r"C:\Users\Robin\Downloads\aki"), metric_type="AUC", include_unfinished=False)
