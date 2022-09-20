source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l files/pretrained_weights/LGBM/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --depth 4 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \

