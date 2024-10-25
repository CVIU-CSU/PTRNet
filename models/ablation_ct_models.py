from torch import nn
import torch
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


class image_model_add_conv(nn.Module):
    def __init__(self, d_model=256, hidden_dim=64):
        super(image_model_add_conv, self).__init__()
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
        img_data = img_data.squeeze(0).permute(0,3,1,2) # N C 14 14

        loss_extra = torch.tensor(0.0).to(img_data.device)

        # soft weight
        slice_feats = torch.nn.functional.interpolate(img_data.unsqueeze(0).permute(0,2,1,3,4), size=(128,14,14), scale_factor=None, mode='trilinear', align_corners=None) # (BCNHW)  / (BCHW) (BCW)
        feat = self.conv3(slice_feats)
        final_feat = self.gap(feat).squeeze(-1).squeeze(-1).squeeze(-1)

        class_output = self.classifier(final_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra
    

class image_model_none(nn.Module):
    def __init__(self, d_model=256, hidden_dim=64):
        super(image_model_none, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(768,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test'):
        img_data = img_data.squeeze(0).permute(0,3,1,2) # N C 14 14

        loss_extra = torch.tensor(0.0).to(img_data.device)

        # soft weight
        slice_feats = torch.nn.functional.interpolate(img_data.unsqueeze(0).permute(0,2,1,3,4), size=(128,14,14), scale_factor=None, mode='trilinear', align_corners=None) # (BCNHW)  / (BCHW) (BCW)
        final_feat = self.gap(slice_feats).squeeze(-1).squeeze(-1).squeeze(-1)

        class_output = self.classifier(final_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra
    


class image_model_add_reembed(nn.Module):
    def __init__(self, d_model=256, hidden_dim=64):
        super(image_model_add_reembed, self).__init__()
        self.atten_pool = FCN()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(768,hidden_dim),
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
        final_feat = self.gap(slice_feats).squeeze(-1).squeeze(-1).squeeze(-1)

        class_output = self.classifier(final_feat)

        if mode != 'train':
            return class_output
        return class_output, loss_extra