source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LGBM.gin \
                             -l files/pretrained_weights/LGBM/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             --depth 7 \
                             --subsample-feat 0.33 \
                             --subsample-data 1.00 \

