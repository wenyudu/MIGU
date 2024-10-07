ini_thresholds=(0.7)
model="t5_large"
method="migu"
# seeds=(42 1024 2048)
seeds=(42)
tuning_method="full_tuning"

for ini_threshold in "${ini_thresholds[@]}"; do
  for seed in "${seeds[@]}"; do

    mkdir -p output/${model}/${tuning_method}/${method}/order_1/logs/${seed}
    LOGFILE="output/${model}/${tuning_method}/${method}/order_1/logs/${seed}/train_and_infer_${ini_threshold}.log"
    if [ ! -f "$LOGFILE" ]; then
        bash scripts/full_tuning/order_1.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} > "$LOGFILE" 2>&1
    else
        echo "Log file already exists: $LOGFILE"
    fi

    # mkdir -p output/${model}/${tuning_method}/${method}/order_2/logs/${seed}
    # LOGFILE="output/${model}/${tuning_method}/${method}/order_2/logs/${seed}/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    #     bash scripts/full_tuning/order_2.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

    # mkdir -p output/${model}/${tuning_method}/${method}/order_3/logs/${seed}
    # LOGFILE="output/${model}/${tuning_method}/${method}/order_3/logs/${seed}/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    #     bash scripts/full_tuning/order_3.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

  done
done