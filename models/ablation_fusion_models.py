from tab_transformer_pytorch import FTTransformer
from tab_transformer_pytorch import TabTransformer
from torch import optim, nn
import torch
import torch.nn.functional as F
from utils.dice_score import dice_loss


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        
        x = self.conv(x)
        x = x.sigmoid()
        return x

class union_model(nn.Module):
    def __init__(self, d_model=256):
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

        self.atten_pool = FCN()
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




class union_model_concat(nn.Module):
    def __init__(self, d_model=256):
        super(union_model_concat, self).__init__()
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

        self.atten_pool = FCN()
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
        union_feat = torch.cat((final_feat, tabular_feat), dim=-1)

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class union_model_add(nn.Module):
    def __init__(self, d_model=256):
        super(union_model_add, self).__init__()
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

        self.atten_pool = FCN()
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Linear(d_model, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        union_feat = final_feat + tabular_feat

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class union_model_add_plus(nn.Module):
    def __init__(self, method='weighted', d_model=256): # 'weighted', 'learned', 'hadamard product'
        super(union_model_add_plus, self).__init__()
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

        self.atten_pool = FCN()
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Linear(d_model, 2)

        self.method = method
        if self.method == 'learned':
            self.w1 = nn.Parameter(torch.tensor(0.5))
            self.w2 = nn.Parameter(torch.tensor(0.5))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        if self.method == 'weighted':
            union_feat = final_feat * 0.465 + tabular_feat * 0.535
        elif self.method == 'learned':
            union_feat = final_feat * F.sigmoid(self.w1)  + tabular_feat * F.sigmoid(self.w2)
        elif self.method == 'hadamard':
            union_feat = final_feat * tabular_feat

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra






class union_model_Film(nn.Module):
    def __init__(self, d_model=256):
        super(union_model_Film, self).__init__()
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

        self.atten_pool = FCN()
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fusion = FiLM(input_dim=256, dim=256, output_dim=256)

        self.classifier = nn.Linear(d_model, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        _,_,union_feat = self.fusion(final_feat, tabular_feat)

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class union_model_gated(nn.Module):
    def __init__(self, d_model=256):
        super(union_model_gated, self).__init__()
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

        self.atten_pool = FCN()
        self.conv3 = nn.Sequential(
            nn.Conv3d(768, 8, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
            nn.Conv3d(8, d_model, kernel_size=(4,3,3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(4,2,2)),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fusion = GatedFusion(input_dim=256, dim=256, output_dim=256)

        self.classifier = nn.Linear(d_model, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        _,_,union_feat = self.fusion(final_feat, tabular_feat)

        class_output = self.classifier(union_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return x, y, output