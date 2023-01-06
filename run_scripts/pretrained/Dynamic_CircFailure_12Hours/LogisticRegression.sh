source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l files/pretrained_weights/LogisticRegression/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --penalty 'l2' \
                             --c_parameter 0.01 \

