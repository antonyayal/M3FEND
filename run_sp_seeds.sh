#!/usr/bin/env bash
# Corre M3FEND en el corpus en espanol: variantes full y zeros x 3 semillas, en secuencia
# (una sola GPU). Cada corrida guarda su log y su checkpoint por separado.
cd "$(dirname "$0")"
mkdir -p logs/sp

for variant in full zeros; do
  for seed in 2021 2022 2023; do
    run="m3fend_sp_${variant}_seed${seed}"
    echo "[$(date '+%F %T')] start $run"
    .venv/bin/python main.py --model_name m3fend --dataset sp --sp_variant "$variant" \
      --lr 0.0001 --seed "$seed" \
      --save_param_dir "./param_model/sp/$run" \
      --param_log_dir "./logs/sp/$run" \
      > "logs/sp/$run.log" 2>&1
    echo "[$(date '+%F %T')] end $run (exit $?)"
  done
done
