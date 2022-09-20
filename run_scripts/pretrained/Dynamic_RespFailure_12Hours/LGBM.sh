source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l files/pretrained_weights/LGBM/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --depth 7 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \

