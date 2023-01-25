import csv

rawNamesMap = {
  "target": "Target",
  "aumc": "AUMCdb",
  "eicu": "eICU",
  "hirid": "HiRID",
  "miiv": "MIMIC-IV",
  "convex_combination_without_target": "Convex UDA",
  "max_prediction": "Max Pooling",
  "target_weight_0.5": "Weighted $\\alpha=1/3$",
  "target_weight_2": "Weighted $\\alpha=2/3$",
  "loss_weighted": "Weighted Loss",
  "bayes_opt": "Weighted Bayes",
  "target_with_predictions": "Prediction-Feature",
  "cc_with_preds": "Combined",
}

def csv_to_dict(file_name):
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    tables = {}
    for row in data:
        row_without_target = {key: value for key, value in row.items() if key != 'target' and key != 'target_size' and key != 'model'}
        tables.setdefault((row['target'], row['target_size']), {})[row['model']] = row_without_target
    return tables


def dict_to_latex(combination, data, metric):
    table = '\\begin{table}[h]\n'
    table += '\\centering\n'
    table += '\\footnotesize'
    table += '\\caption{{Sepsis prediction on {0} with target size {1}, {2} with standard deviation.}}\n'.format(rawNamesMap[combination[0]], combination[1], "AUROC" if metric == "auc" else "AUPRC")
    headers = ['Model']
    for model, scores in data.items():
        headers += [model]

    table += '\\begin{tabular}{l|' + ''.join(['c'] * (len(headers) - 1)) + '}\n'
    table += '\\textbf{' + '} & \\textbf{'.join(headers) + '}\\\\\n'
    table += '\\hline\n'

    for score_name, score in data[model].items():
        if "_avg" in score_name:
            raw_name = score_name.split("_avg")[0]
            if raw_name == combination[0] or not raw_name in rawNamesMap:
                continue
            clean_name = rawNamesMap[raw_name]
            values = [clean_name]
            for model in headers[1:]:
                scores = data[model]
                avg = "{:.2f}".format(float(scores[score_name]))
                std = "{:.2f}".format(float(scores[f"{raw_name}_std"]))
                values.append(f"${avg} \pm {std}$")
            table += ' & '.join(values) + '\\\\\n'

    table += '\\end{tabular}\n'
    table += '\\end{table}\n'
    return table

if __name__ == '__main__':
    for metric in ["auc", "pr"]:
        file_name = f'../yaib_logs/sep_{metric}.csv'
        data = csv_to_dict(file_name)
        for key, row in data.items():
            table = dict_to_latex(key, row, metric)
            print(table)
        print('\n' * 5)
        
