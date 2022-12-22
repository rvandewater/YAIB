import json
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import logging
import gin
from pathlib import Path

import pandas as pd



log_parent = Path(r"C:\Users\Robin\Downloads\yaib_experiments")
aggregated = pd.DataFrame(columns=["Dataset","Model", "Avg", "Std", "95% CI"])
# pd.MultiIndex.from_tuples(aggregated, names=[("Dataset","Model")])
metric_type = "PR"
accumulation = "avg"
for dataset in log_parent.iterdir():
    for task in dataset.iterdir():
        for model in task.iterdir():
            for log_time in model.iterdir():

                if(Path(log_time / "accumulated_test_metrics.json").exists()):
                    test_metrics = json.load(open(log_time / "accumulated_test_metrics.json"))
                    avg = test_metrics[accumulation][metric_type]
                    std = test_metrics["std"][metric_type]
                    ci_95 = test_metrics["CI_0.95"][metric_type]
                else:
                    avg = "NaN"
                    std = "NaN"
                    ci_95 = "NaN"
                aggregated.loc[len(aggregated.index)] = [dataset.name, model.name, avg, std, ci_95]
                # print(aggregated[dataset.name, model.name])
                # aggregated = pd.concat([aggregated, pd.DataFrame({"Dataset": dataset.name, "Model": model.name})], ignore_index=True)
                    # = json.load(open(log_time / "accumulated_test_metrics.json"))
# aggregated = aggregated.sort_values(by=["Model"])
aggregated = aggregated.round(2)
pd.to_numeric(aggregated["Avg"])
print(aggregated.round(2).to_markdown())