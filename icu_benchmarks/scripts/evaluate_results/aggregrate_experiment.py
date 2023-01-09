import json
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import logging
import gin
import math
from pathlib import Path

import pandas as pd



log_parent = Path(r"C:\Users\Robin\Downloads\yaib_experiments_aki")
aggregated = pd.DataFrame(columns=["Dataset","Model", "Avg", "Std", "95% CI"])
# pd.MultiIndex.from_tuples(aggregated, names=[("Dataset","Model")])
metric_type = "PR"
accumulation = "avg"
seeds = 5
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
nan_rows = aggregated[aggregated["95% CI"]=="NaN"]
aggregated = aggregated[aggregated["95% CI"]!="NaN"]

aggregated["Avg"] = pd.to_numeric(aggregated["Avg"], errors="coerce")
aggregated["Std"] = pd.to_numeric(aggregated["Std"], errors="coerce")
aggregated["95% CI"] = aggregated["95% CI"].apply(lambda x: (pd.to_numeric(x[1])*100, pd.to_numeric(x[0])*100))
aggregated[["Avg", "Std"]] = aggregated[["Avg", "Std"]].apply(lambda x: x*100)
aggregated["Std"] = aggregated["Std"].apply(lambda x: x/math.sqrt(seeds))
# aggregated[""] = aggregated.apply(pd.to_numeric, columns= ["Avg", "Std"], errors='coerce')
aggregated = aggregated.round(2)
aggregated["95% CI"] = aggregated["95% CI"].apply(lambda x: tuple(map(lambda y: round(y,2),x)))
aggregated = aggregated.append(nan_rows)
aggregated = aggregated.set_index(["Dataset"])
# aggregated = aggregated.reset_index(drop=True)
# pd.to_numeric(aggregated["Avg"])
aggregated = aggregated.sort_values(by=["Dataset", "Model"], ascending=[True, True])
print(aggregated.to_markdown())