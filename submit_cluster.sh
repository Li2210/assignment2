spark-submit \
    --master yarn \
    --deploy-mode cluster \
    MovieData_Summary.py \
    --output $1
