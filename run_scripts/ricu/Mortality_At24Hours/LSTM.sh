source activate yaib_updated
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LSTM.gin \
                             -l logs/benchmark_exp/LSTM/ \
                             -t Mortality_At24Hours \
                             -o True \
                             -lr 1e-4\
                             --hidden 128 \
                             --do 0.1 \
                             --depth 1 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

