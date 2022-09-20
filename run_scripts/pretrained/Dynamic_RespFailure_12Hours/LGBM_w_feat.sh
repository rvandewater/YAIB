source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --depth 6 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.66 \

