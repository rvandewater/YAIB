source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             --depth 7 \
                             --subsample-feat 0.66 \
                             --subsample-data 0.33 \

