python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm vit --tab tabtransformer --reembed 0
python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm vit --tab tabtransformer --reembed 1

python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm clip --tab tabtransformer --reembed 0
python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm clip --tab tabtransformer --reembed 1

python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm biomedclip --tab tabtransformer --reembed 0
python scripts/train_reembed_ablation.py --saveName reembed_ablation --d_model 256 --vfm biomedclip --tab tabtransformer --reembed 1
# results: log/reembed_ablation

python scripts/train_dinov2.py --saveName dinov2 --d_model 256 --vfm dinov2 --tab tabtransformer --reembed 0
python scripts/train_dinov2.py --saveName dinov2 --d_model 256 --vfm dinov2 --tab tabtransformer --reembed 1
# results: log/dinov2_ablation