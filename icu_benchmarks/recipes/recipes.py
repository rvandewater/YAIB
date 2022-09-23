import pandas as pd

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.step import StepImputeFill, StepScale, StepHistorical

if __name__ == "__main__":
    df = pd.read_csv('/Users/patrick/datasets/benchmark/sepsis/mimic/dyn.csv.gz', compression='gzip')
    df = df[['stay_id', 'time', 'hr', 'resp', 'temp', 'sbp', 'dbp', 'map']]

    rec = Recipe(df)
    rec.add_role('stay_id', 'group')
    rec.add_role(['hr', 'resp', 'temp', 'sbp', 'dbp', 'map'], 'predictor')

    rec.add_step(StepScale())
    rec.add_step(StepHistorical(fun='max'))
    rec.add_step(StepImputeFill(method='ffill'))
    rec.add_step(StepImputeFill(value=0))
    
    rec.prep()
    rec.bake()

