source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/LGBM.gin \
                             -l files/pretrained_weights/LGBM/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             --depth 4 \
                             --subsample-feat 1.0 \
                             --subsample-data 1.00 \

