from tab_transformer_pytorch import FTTransformer
from tab_transformer_pytorch import TabTransformer
from torch import optim, nn
import torch
import torch.nn.functional as F
from utils.dice_score import dice_loss


class FCN(nn.Module):
    def __init__(self, layer_num = 4):
        super(FCN, self).__init__()
        
        if layer_num == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
        elif layer_num == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
        elif layer_num == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
        elif layer_num == 4:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        elif layer_num == 5:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        elif layer_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        
    def forward(self, x):
        
        x = self.conv(x)
        x = x.sigmoid()
        return x

class image_model(nn.Module):
    def __init__(self, d_model=256, hidden_dim=64):
        super(image_model, self).__init__()
        self.atten_pool = FCN()
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(d_model,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test'):
        B = img_data.shape[1]
        img_data = img_data.squeeze(0).permute(0,3,1,2) # B C 14 14

        pred_mask = self.atten_pool(img_data).squeeze(1)
        gt_mask = inner_slice_mask[0,:,1:].reshape(B,14,14)

        if mode == 'vis':
            return pred_mask
        loss_extra = dice_loss(pred_mask, gt_mask)

        # soft weight
        slices_feat = (img_data * pred_mask.unsqueeze(1)).unsqueeze(0).permute(0,2,1,3,4)
        slice_feats = torch.nn.functional.interpolate(slices_feat, size=(128,14,14), scale_factor=None, mode='trilinear', align_corners=None)
        feat = self.conv3(slice_feats)
        final_feat = self.gap(feat).squeeze(-1).squeeze(-1).squeeze(-1)

        class_output = self.classifier(final_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class union_model(nn.Module):
    def __init__(self, d_model=256, layer_num=4):
        super(union_model, self).__init__()
        self.tabular_transformer = TabTransformer(
                            categories = (3,2,2,2),      # tuple containing the number of unique values within each category
                            num_continuous = 23,                # number of continuous values
                            dim = 32,                           # dimension, paper set at 32
                            dim_out = d_model,                        # binary prediction, but could be anything
                            depth = 6,                          # depth, paper recommended 6
                            heads = 8,                          # heads, paper recommends 8
                            attn_dropout = 0.1,                 # post-attention dropout
                            ff_dropout = 0.1,                   # feed forward dropout
                            mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                            mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
                            continuous_mean_std = None          # (optional) - normalize the continuous values before layer norm
                            )

        self.atten_pool = FCN(layer_num=layer_num)
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Linear(d_model*2, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

        self.fusion = MMTM(d_model,d_model,1.0)

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test'):

        tabular_feat = self.tabular_transformer(x_categ, x_numer)

        B = img_data.shape[1]
        img_data = img_data.squeeze(0).permute(0,3,1,2) # B C 14 14

        pred_mask = self.atten_pool(img_data).squeeze(1)
        gt_mask = inner_slice_mask[0,:,1:].reshape(B,14,14)

        if mode == 'vis':
            return pred_mask
        loss_extra = dice_loss(pred_mask, gt_mask)

        # soft weight
        slices_feat = (img_data * pred_mask.unsqueeze(1)).unsqueeze(0).permute(0,2,1,3,4)

        slice_feats = torch.nn.functional.interpolate(slices_feat, size=(128,14,14), scale_factor=None, mode='trilinear', align_corners=None)
        feat = self.conv3(slice_feats)
        final_feat = self.gap(feat).squeeze(-1).squeeze(-1).squeeze(-1)

        # fusion
        union_feat = self.fusion(final_feat, tabular_feat)

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class tabular_model(nn.Module):
    def __init__(self, d_model=256, hidden_dim=64):
        super(tabular_model, self).__init__()
        self.tabular_transformer = TabTransformer(
                            categories = (3,2,2,2),      # tuple containing the number of unique values within each category
                            num_continuous = 23,                # number of continuous values
                            dim = 32,                           # dimension, paper set at 32
                            dim_out = d_model,                        # binary prediction, but could be anything
                            depth = 6,                          # depth, paper recommended 6
                            heads = 8,                          # heads, paper recommends 8
                            attn_dropout = 0.1,                 # post-attention dropout
                            ff_dropout = 0.1,                   # feed forward dropout
                            mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                            mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
                            continuous_mean_std = None          # (optional) - normalize the continuous values before layer norm
                            )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model,hidden_dim),
            nn.Linear(hidden_dim,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test'):

        tabular_feat = self.tabular_transformer(x_categ, x_numer)
        class_output = self.classifier(tabular_feat)
        return class_output

class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, skeleton):
        squeeze = torch.cat((visual, skeleton),dim=-1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        vis_out = vis_out.unsqueeze(1).repeat(1,visual.shape[1],1)
        vis_res = (visual * vis_out).mean(1)

        sk_out = sk_out.unsqueeze(1).repeat(1,skeleton.shape[1],1)
        sk_res = (skeleton * sk_out).mean(1)

        return torch.cat((vis_res, sk_res), dim=-1)