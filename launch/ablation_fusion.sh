python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode add --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode concat --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode weighted --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode learned --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode hadamard --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode gated --d_model 256
python scripts/train_fusion_ablation.py --saveName fusion_ablation --model_mode film --d_model 256
# results: log/fusion_ablation