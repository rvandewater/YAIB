source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l files/pretrained_weights/LGBM/ \
                             -t Phenotyping_APACHEGroup \
                             --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             --depth 3 \
                             --subsample-feat 0.66 \
                             --subsample-data 0.66 \


