spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 4 \
    --executor-memory 8G \
    --executor-cores 4 \
    assignment2.py \
    --output $1
