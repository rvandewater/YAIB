import json
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import logging
import gin
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
    sort=["Dataset", "Model"],
    datasets=["miiv", "aumc", "hirid", "eicu"],
):
    results = pd.DataFrame(columns=["Time", "Dataset", "Model", "Average", "Std", "95% CI", "Execution Time"])
    for dataset in log_dir.iterdir():
        if dataset.is_dir() and dataset.name in datasets:
            for task in dataset.iterdir():
                for model in task.iterdir():
                    for log_time in model.iterdir():
                        if Path(log_time / "accumulated_test_metrics.json").exists():
                            test_metrics = json.load(open(log_time / "accumulated_test_metrics.json"))
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

    nan_rows = results[results["95% CI"] == "NaN"]
    results = results[results["95% CI"] != "NaN"]
    #results["Time"] = results["Time"].apply(lambda x: pd.to_datetime(x, format="%H:%M:%S"))
    results["Execution Time"] = pd.to_timedelta(results["Execution Time"])
    results["Average"] = pd.to_numeric(results["Average"], errors="coerce")
    results["Std"] = pd.to_numeric(results["Std"], errors="coerce")
    results["95% CI"] = results["95% CI"].apply(lambda x: (pd.to_numeric(x[1]) * 100, pd.to_numeric(x[0]) * 100))
    results[["Average", "Std"]] = results[["Average", "Std"]].apply(lambda x: x * 100)
    # Correct std for amount of iterations
    if iterations > 1:
        results["Std"] = results["Std"].apply(lambda x: x / math.sqrt(iterations))

    # Round everything to x decimals
    results = results.round(decimals)
    results["95% CI"] = results["95% CI"].apply(lambda x: tuple(map(lambda y: round(y, decimals), x)))

    # Append unfinished results
    results = results.append(nan_rows)
    results = results.set_index(["Dataset"])
    results = results.sort_values(by=sort, ascending=[True, True])
    results = results[results["Model"].isin(models)]
    print(results.to_markdown())


aggregate_results(Path(r"C:\Users\Robin\Downloads\aki"), metric_type="AUC", include_unfinished=False)
