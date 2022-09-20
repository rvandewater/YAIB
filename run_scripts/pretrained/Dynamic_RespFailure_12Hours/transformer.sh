source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.0 \
                             --do_att 0.3 \
                             --depth 2 \
                             --heads 1 \


