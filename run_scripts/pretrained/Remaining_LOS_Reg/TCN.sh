source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/TCN.gin \
                             -l files/pretrained_weights/TCN/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             --hidden 128 \
                             -lr 3e-4\
                             --do 0.3 \
                             --kernel 32 \


