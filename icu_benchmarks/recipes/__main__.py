import pandas as pd
from pyarrow import parquet
from sklearn.preprocessing import *
from sklearn.impute import SimpleImputer

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.step import StepImputeFill, StepScale, StepHistorical, StepSKlearn
from icu_benchmarks.recipes.selector import all_numeric_predictors

if __name__ == "__main__":
    df = parquet.read_table("/Users/hendrikschmidt/projects/thesis/data/ricu/mimic/dyn.parquet").to_pandas()
    df = df[['stay_id', 'time', 'hr', 'resp', 'temp', 'sbp', 'dbp', 'map']]

    rec = Recipe(df)
    rec.add_role('stay_id', 'group')
    rec.add_role(['hr', 'resp', 'temp', 'sbp', 'dbp', 'map'], 'predictor')

    # rec.add_step(StepScale())
    # rec.add_step(StepHistorical(fun='max'))
    # rec.add_step(StepImputeFill(method='ffill'))
    # rec.add_step(StepImputeFill(value=0))
    rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=SimpleImputer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=StandardScaler()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=MinMaxScaler()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=RobustScaler()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=MaxAbsScaler()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=QuantileTransformer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=PowerTransformer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=Normalizer()))
    rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=KBinsDiscretizer(encode='onehot-dense'), is_in_place=False))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=Normalizer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=Normalizer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=Normalizer()))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=LabelBinarizer(), is_in_place=False, columnwise=True))
    # rec.add_step(StepSKlearn(sel=all_numeric_predictors(), sklearn_transform=OneHotEncoder(), is_in_place=False))

    print(rec)
    

    df_i = rec.prep(df.iloc[:-10000, :])
    rec.bake(df.iloc[10000:, :])

    print(df.iloc[:-10000, :])
    print(df_i)
