source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Phenotyping_APACHEGroup \
                             --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             --depth 5 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \


