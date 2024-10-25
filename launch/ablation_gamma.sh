python scripts/train_gamma_ablation.py --saveName gamma_ablation --model_mode union --d_model 256 --gamma 0.1
python scripts/train_gamma_ablation.py --saveName gamma_ablation --model_mode union --d_model 256 --gamma 0.2
python scripts/train_gamma_ablation.py --saveName gamma_ablation --model_mode union --d_model 256 --gamma 0.3
python scripts/train_gamma_ablation.py --saveName gamma_ablation --model_mode union --d_model 256 --gamma 0.4
python scripts/train_gamma_ablation.py --saveName gamma_ablation --model_mode union --d_model 256 --gamma 0.5
# results: log/gamma_ablation

python scripts/train_no_strategy.py
# results: log/no_strategy