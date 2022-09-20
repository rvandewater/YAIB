source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l files/pretrained_weights/LogisticRegression/ \
                             -t Phenotyping_APACHEGroup \
                             --loss-weight balanced \
                             --maxlen 288 \
                             --c_parameter 0.1\
			                 --penalty 'l2' \

