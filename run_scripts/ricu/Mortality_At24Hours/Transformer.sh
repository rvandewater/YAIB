source activate yaib_updated
python -m icu_benchmarks.run train \
                             -c configs/ricu/Classification/transformer.gin \
                             -l logs/benchmark_exp/transformer/ \
                             -t Mortality_At24Hours\
                             -o True \
                             -lr 1e-5 \
                             --hidden 256 \
                             --do 0.1 \
                             --depth 2 \
                             --heads 4 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

