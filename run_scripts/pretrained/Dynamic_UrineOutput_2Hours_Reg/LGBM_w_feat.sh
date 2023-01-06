source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             --depth 6 \
                             --subsample-feat 1.0 \
                             --subsample-data 1.00 \

