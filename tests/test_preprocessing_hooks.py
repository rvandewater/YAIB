import pytest
import icu_benchmarks.run as run
import gin

def test_preprocessing_hooks():
    run("train -d demo_data/mortality24/mimic_demo -n mimic_demo -t BinaryClassification -tn Mortality24 -m LGBMClassifier -s 2222 -l ../yaib_logs/ --tune -p configs/preprocessing/preprocessor_test.py")
    assert gin.query_parameter("preprocess.preprocessor")  == "CustomPreprocessor.CustomPreprocessor"