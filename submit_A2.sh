spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 4 \
    --executor-cores 4 \
    --conf spark.default.parallelism = 64 \
    assignment2.py \
    --output $1
