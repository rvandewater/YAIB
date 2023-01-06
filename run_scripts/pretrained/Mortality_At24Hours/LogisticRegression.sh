source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l files/pretrained_weights/LogisticRegression/ \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -o True \
                             --c_parameter 1\
			                 --penalty 'l2' \


