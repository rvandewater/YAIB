source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             -lr 3e-5\
                             -bs 8\
                             --hidden 128 \
                             --do 0.0 \
                             --do_att 0.4\
                             --depth 3 \
                             --heads 1 \


