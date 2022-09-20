source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/GRU.gin \
                             -l files/pretrained_weights/GRU/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             -lr 3e-4\
                             --hidden 128 \
                             --do 0.3 \
                             --depth 2 \


