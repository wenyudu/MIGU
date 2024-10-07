ßcluster=1024
ini_thresholds=(0.7)
cluster_constructure_method="sequential"
activation_combined=True
method="cluster_activate"

seeds=(42 1024 2048)

for ini_threshold in "${ini_thresholds[@]}"; do
  for seed in "${seeds[@]}"; do

    mkdir -p output/t5_large/${method}/sequential/order_6/logs/${seed}
    LOGFILE="output/t5_large/${method}/sequential/order_6/logs/${seed}/train_and_infer_${ini_threshold}.log"
    if [ ! -f "$LOGFILE" ]; then
        bash scripts/t5_large/${method}/order_6.sh ${cluster} ${ini_threshold} ${cluster_constructure_method} ${activation_combined} ${seed} ${method} > "$LOGFILE" 2>&1
    else
        echo "Log file already exists: $LOGFILE"
    fi

  done
done