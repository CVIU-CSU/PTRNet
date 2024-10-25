python scripts/train_ct_ablation.py --saveName ct_ablation --model_mode all --d_model 256
python scripts/train_ct_ablation.py --saveName ct_ablation --model_mode reembed --d_model 256
python scripts/train_ct_ablation.py --saveName ct_ablation --model_mode conv --d_model 256
python scripts/train_ct_ablation.py --saveName ct_ablation --model_mode none --d_model 256
# results: log/ct_ablation