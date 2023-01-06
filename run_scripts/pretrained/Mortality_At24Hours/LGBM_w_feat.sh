source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -o True \
                             --depth 7 \
                             --subsample-feat 1.00 \
                             --subsample-data 1.00 \


