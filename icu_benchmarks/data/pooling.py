from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import DataSegment as Segment, VarType as Var
from icu_benchmarks.contants import RunMode
import pyarrow.parquet as pq


class PooledDataset:
    hirid_eicu_miiv = ["hirid", "eicu", "miiv"]
    aumc_hirid_eicu = ["aumc", "hirid", "eicu"]
    aumc_eicu_miiv = ["aumc", "eicu", "miiv"]
    aumc_hirid_miiv = ["aumc", "hirid", "miiv"]
    aumc_hirid_eicu_miiv = ["aumc", "hirid", "eicu", "miiv"]


class PooledData:
    def __init__(
        self,
        data_dir,
        vars,
        datasets,
        file_names,
        shuffle=False,
        stratify=None,
        runmode=RunMode.classification,
        save_test=True,
    ):
        """
        Generate pooled data from existing datasets.
        Args:
            data_dir: Where to read the data from
            vars: Variables dictionary
            datasets: Which datasets to pool
            file_names: Which files to read from
            shuffle: Whether to shuffle data
            stratify: Stratify data
            runmode: Which task runmode
            save_test: Save left over test data to test on without leakage
        """
        self.data_dir = data_dir
        self.vars = vars
        self.datasets = datasets
        self.file_names = file_names
        self.shuffle = shuffle
        self.stratify = stratify
        self.runmode = runmode
        self.save_test = save_test

    def generate(
        self,
        datasets,
        samples=10000,
        seed=42,
    ):
        """
        Generate pooled data from existing datasets.
        Args:
            datasets: Which datasets to pool
            samples: Amount of samples to pool
            seed: Random seed
        """
        data = {}
        for folder in self.data_dir.iterdir():
            if folder.is_dir():
                if folder.name in datasets:
                    data[folder.name] = {
                        f: pq.read_table(folder / self.file_names[f]).to_pandas(self_destruct=True)
                        for f in self.file_names.keys()
                    }
        data = self._pool_datasets(
            datasets=data,
            samples=samples,
            vars=vars,
            shuffle=self.shuffle,
            stratify=self.stratify,
            seed=seed,
            runmode=self.runmode,
            data_dir=self.data_dir,
            save_test=self.save_test,
        )
        self._save_pooled_data(self.data_dir, data, datasets, self.file_names, samples=samples)

    def _save_pooled_data(self, data_dir, data, datasets, file_names, samples=10000):
        """
        Save pooled data to disk.
        Args:
            data_dir: Directory to save the data
            data: Data to save
            datasets: Which datasets were pooled
            file_names: The file names to save to
            samples: Amount of samples to save
        """
        save_folder = "_".join(datasets)
        save_folder += f"_{samples}"
        save_dir = data_dir / save_folder
        if not save_dir.exists():
            save_dir.mkdir()
        for key, value in data.items():
            value.to_parquet(save_dir / Path(file_names[key]))
        logging.info(f"Saved pooled data at {save_dir}")

    def _pool_datasets(
        self,
        datasets={},
        samples=10000,
        vars=[],
        seed=42,
        shuffle=True,
        runmode=RunMode.classification,
        data_dir=Path("data"),
        save_test=True,
    ):
        """
        Pool datasets into a single dataset.
        Args:
            datasets: list of datasets to pool
            samples: Amount of samples
            vars: The variables dictionary
            seed: Random seed
            shuffle: Shuffle samples
            runmode: Runmode
            data_dir: Where to save the data
            save_test: If true, save test data to test on without leakage
        Returns:
            pooled dataset
        """
        if len(datasets) == 0:
            raise ValueError("No datasets supplied.")
        pooled_data = {Segment.static: [], Segment.dynamic: [], Segment.outcome: []}
        id = vars[Var.group]
        int_id = 0
        for key, value in datasets.items():
            int_id += 1
            # Preventing id clashing
            repeated_digit = str(int_id) * 4
            outcome = value[Segment.outcome]
            static = value[Segment.static]
            dynamic = value[Segment.dynamic]
            # Get unique stay IDs from outcome segment
            stays = pd.Series(outcome[id].unique())

            if runmode is RunMode.classification:
                # If we have more outcomes than stays, check max label value per stay id
                labels = outcome.groupby(id).max()[vars[Var.label]].reset_index(drop=True)
                # if pd.Series(outcome[id].unique()) is outcome[id]):
                selected_stays = train_test_split(
                    stays, stratify=labels, shuffle=shuffle, random_state=seed, train_size=samples
                )
            else:
                selected_stays = train_test_split(stays, shuffle=shuffle, random_state=seed, train_size=samples)
            # Select only stays that are in the selected_stays
            # Save test sets to test on without leakage
            if save_test:
                select = selected_stays[1]
                outcome, static, dynamic = self._select_stays(
                    outcome=outcome, static=static, dynamic=dynamic, select=select, repeated_digit=repeated_digit
                )
                save_folder = key
                save_folder += f"_test_{len(select)}"
                save_dir = data_dir / save_folder
                if not save_dir.exists():
                    save_dir.mkdir()
                outcome.to_parquet(save_dir / Path("outc.parquet"))
                static.to_parquet(save_dir / Path("sta.parquet"))
                dynamic.to_parquet(save_dir / Path("dyn.parquet"))
                logging.info(f"Saved train data at {save_dir}")
            selected_stays = selected_stays[0]
            outcome, static, dynamic = self._select_stays(
                outcome=outcome, static=static, dynamic=dynamic, select=selected_stays, repeated_digit=repeated_digit
            )
            # Adding to pooled data
            pooled_data[Segment.static].append(static)
            pooled_data[Segment.dynamic].append(dynamic)
            pooled_data[Segment.outcome].append(outcome)
        # Add each datatype together
        for key, value in pooled_data.items():
            pooled_data[key] = pd.concat(value, ignore_index=True)
        return pooled_data

    def _select_stays(self, outcome, static, dynamic, select, repeated_digit=1):
        """Selects stays for outcome, static, dynamic dataframes.

        Args:
            outcome: Outcome dataframe
            static: Static dataframe
            dynamic: Dynamic dataframe
            select: Stay IDs to select
            repeated_digit: Digit to repeat for ID clashing
        """
        outcome = outcome.loc[outcome[id].isin(select)]
        static = static.loc[static[id].isin(select)]
        dynamic = dynamic.loc[dynamic[id].isin(select)]
        # Preventing id clashing
        outcome[id] = outcome[id].map(lambda x: int(str(x) + repeated_digit))
        static[id] = static[id].map(lambda x: int(str(x) + repeated_digit))
        dynamic[id] = dynamic[id].map(lambda x: int(str(x) + repeated_digit))
        return outcome, static, dynamic
