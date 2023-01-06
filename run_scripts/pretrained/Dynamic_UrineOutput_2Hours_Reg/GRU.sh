source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/GRU.gin \
                             -l files/pretrained_weights/GRU/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             -lr 3e-4\
                             --hidden 256 \
                             --do 0.3 \
                             --depth 3 \


