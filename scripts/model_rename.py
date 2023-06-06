import os
import shutil

def rename_and_move_files(folder_path, destination_folder):
    for root, fold, files in os.walk(os.path.join(os.path.join(folder_path,"repetition_0"),"fold_0")):
        for filename in files:
                if(filename.endswith(".torch") or filename.endswith(".joblib") or filename.endswith(".ckpt") or filename.endswith(".txt")):
                    subfolder_name = os.path.basename(root)
                    parent_folder_name = os.path.basename(os.path.dirname(root))
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(root))))
                    task_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(root)))))
                    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(root))))))
                    old_path = os.path.join(root, filename)
                    new_filename = f"{dataset_name}_kf_{model_name}_{parent_folder_name}_{subfolder_name}_{filename}"
                    new_folder = os.path.join(destination_folder, task_name, dataset_name)
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    new_path = os.path.join(new_folder, new_filename)
                    shutil.copy2(old_path, new_path)
                    print(f"Moved: {old_path} --> {new_path}")

# Example usage:
destination_folder = r"C:\Users\Robin\Downloads\corrected_kf_regression"
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\GRU\2023-01-25T10-42-45", destination_folder)
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\TCN\2023-01-29T13-05-10", destination_folder)
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\LogisticRegression\2023-01-29T13-11-01", destination_folder)
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\LSTM\2023-01-17T10-17-13", destination_folder)
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\LGBMClassifier\2023-01-20T15-50-43", destination_folder)
# rename_and_move_files(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\Transformer\2023-01-27T16-15-05", destination_folder)

def get_most_recent_date(folder_path):
    all_dates = []
    for folder in os.listdir(folder_path):
            folder_date = folder
            if "repetition_0" in os.listdir(os.path.join(folder_path, folder)):
                folder_date = os.path.join(folder_path, folder)
                all_dates.append(folder_date)
    if len(all_dates) != 0:
        most_recent_date = max(all_dates)
    return most_recent_date
task = "Regression"
# for dataset in [rf"C:\Users\Robin\Downloads\yaib results\{task}\miiv\{task}",
#                 rf"C:\Users\Robin\Downloads\yaib results\{task}\eicu\{task}",
#                 rf"C:\Users\Robin\Downloads\yaib results\{task}\hirid\{task}",
#                 rf"C:\Users\Robin\Downloads\yaib results\{task}\aumc\{task}"]:
base_path = r"C:\Users\Robin\Downloads\crea_sweep_el"
for dataset in [rf"{base_path}\miiv\{task}",
                rf"{base_path}\eicu\{task}",
                rf"{base_path}\hirid\{task}",
                rf"{base_path}\aumc\{task}"]:
    base_path = dataset
    for folder in os.listdir(base_path):
        file = os.path.join(base_path,os.path.join(folder, get_most_recent_date(os.path.join(base_path,folder))))
        print(file)
        rename_and_move_files(file, destination_folder)
    # print(file)


# print(get_most_recent_date(r"C:\Users\Robin\Downloads\yaib results\sepsis\miiv\sepsis\TCN"))