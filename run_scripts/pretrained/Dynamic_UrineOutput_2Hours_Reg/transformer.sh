source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Regression/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             -lr 3e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.1 \
                             --do_att 0.1 \
                             --depth 1 \
                             --heads 1 \


