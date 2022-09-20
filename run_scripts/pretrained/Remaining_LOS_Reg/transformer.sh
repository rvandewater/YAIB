source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             -lr 3e-4\
                             -bs 8\
                             --hidden 128 \
                             --do 0.3 \
                             --depth 1 \
                             --heads 1 \


