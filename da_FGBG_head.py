import torch
import torch.nn.functional as F
from torch import nn

from models.da_FGBGloss import DALossComputation


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return gradient_scalar(x, self.weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(weight={self.weight})"


class DAImgHead(nn.Module):
    """
    Image-level Domain Classifier
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        """
        x: list of Tensors, each shape [B, in_channels, H, W]
        """
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))  # shape [B,1,H,W]
        return img_features


class DAInsHead(nn.Module):
    """
    Simple Instance-level Domain Classifier (Fully Connected)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)

        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        # x: [N, in_channels]
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc3_da(x)  # [N, 1]


class DomainAdaptationModule(nn.Module):
    """
    Module for Domain Adaptation.
    We'll do multiple calls to self.loss_evaluator (DALossComputation),
    each time with different (da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels) sets.

    This way we do not modify DALossComputation itself.
    """
    def safe_reshape(self, feat, default_channels, device):
        if feat is not None:
            if feat.numel() == 0:
                return torch.zeros((0, default_channels), device=device)
            else:
                return feat.view(feat.size(0), -1)
        else:
            return torch.zeros((0, default_channels), device=device)

    def __init__(self, DA_HEADS):
        super().__init__()

        self.img_weight = DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = DA_HEADS.DA_CST_LOSS_WEIGHT
        self.lw_da_ins = DA_HEADS.LW_DA_INS

        # GRLs
        self.grl_img = GradientScalarLayer(-1.0 * 0.1)
        self.grl_img_consist = GradientScalarLayer(1.0 * 0.1)

        self.grl_ins_fg_after = GradientScalarLayer(-1.0 * 0.1)
        self.grl_ins_fg_after_cons = GradientScalarLayer(1.0 * 0.1)
        self.grl_ins_fg_before = GradientScalarLayer(-1.0 * 0.1)
        self.grl_ins_fg_before_cons = GradientScalarLayer(1.0 * 0.1)

        self.grl_ins_bg_after = GradientScalarLayer(-1.0 * 0.1)
        self.grl_ins_bg_after_cons = GradientScalarLayer(1.0 * 0.1)
        self.grl_ins_bg_before = GradientScalarLayer(-1.0 * 0.1)
        self.grl_ins_bg_before_cons = GradientScalarLayer(1.0 * 0.1)

        # Heads
        self.imghead = DAImgHead(in_channels=1024)  # for image-level
        self.fg_inshead_after = DAInsHead(in_channels=256)
        self.fg_inshead_before = DAInsHead(in_channels=2048)
        self.bg_inshead_after = DAInsHead(in_channels=256)
        self.bg_inshead_before = DAInsHead(in_channels=2048)

        self.loss_evaluator = DALossComputation()

        self.alpha_bg = 0.3





    def forward(
        self,
        # 图像特征
        img_features,
        # 前景 AFTER
        fg_ins_feat, fg_ins_labels,
        # 背景 AFTER
        bg_ins_feat, bg_ins_labels,
        # 前景 BEFORE
        fg_ins_feat_bf, fg_ins_labels_bf,
        # 背景 BEFORE
        bg_ins_feat_bf, bg_ins_labels_bf,
        targets=None,
    ):
        """
        这8个张量由 roi_heads.extract_da(...) 返回
        """
        if not self.training:
            return {}

        losses = {}


          # =============== (0) 预处理：全部 reshape ===============
        # view 到 [B, C]
  

        # 获取设备信息，假设img_features非空
        device = img_features[0].device if img_features else torch.device('cpu')

        # 处理各个特征，指定对应的默认通道数
        fg_ins_feat = self.safe_reshape(fg_ins_feat, 256, device)
        fg_ins_feat_bf = self.safe_reshape(fg_ins_feat_bf, 2048, device)
        bg_ins_feat = self.safe_reshape(bg_ins_feat, 256, device)
        bg_ins_feat_bf = self.safe_reshape(bg_ins_feat_bf, 2048, device)

        # =============== (A) 先做一次 "图像级 + 前景 AFTER" ===============
        # 1) GRL (image-level)
        img_grl = [self.grl_img(f) for f in img_features]
        da_img_feats = self.imghead(img_grl)

        # 1.1) 前景 AFTER
        fg_after_grl = self.grl_ins_fg_after(fg_ins_feat)
        fg_after_out = self.fg_inshead_after(fg_after_grl)
        # consistency
        img_grl_cst = [self.grl_img_consist(f) for f in img_features]
        da_img_feats_cst = self.imghead(img_grl_cst)
        da_img_feats_cst = [o.sigmoid() for o in da_img_feats_cst]

        fg_after_grl_cst = self.grl_ins_fg_after_cons(fg_ins_feat)
        fg_after_out_cst = self.fg_inshead_after(fg_after_grl_cst).sigmoid()

        # => 调 DALossComputation
        da_img_loss_A, da_fg_after_loss, da_fg_after_cst = self.loss_evaluator(
            da_img=da_img_feats,
            da_ins=fg_after_out,
            da_img_consist=da_img_feats_cst,
            da_ins_consist=fg_after_out_cst,
            da_ins_labels=fg_ins_labels,
            targets=targets
        )

        # =============== (B) 背景 AFTER (不重复图像级) ===============
        # 这里 da_img=None, da_img_consist=None => 不做图像对抗, 只做 BG
        bg_after_grl = self.grl_ins_bg_after(bg_ins_feat)
        bg_after_out = self.bg_inshead_after(bg_after_grl)
        bg_after_grl_cst = self.grl_ins_bg_after_cons(bg_ins_feat)
        bg_after_out_cst = self.bg_inshead_after(bg_after_grl_cst).sigmoid()

        da_img_loss_B, da_bg_after_loss, da_bg_after_cst = self.loss_evaluator(
            da_img=da_img_feats,  # 不做图像
            da_ins=bg_after_out,
            da_img_consist=da_img_feats_cst,
            da_ins_consist=bg_after_out_cst,
            da_ins_labels=bg_ins_labels,
            targets=targets
        )

        # =============== (C) 前景 BEFORE ===============
        fg_before_grl = self.grl_ins_fg_before(fg_ins_feat_bf)
        fg_before_out = self.fg_inshead_before(fg_before_grl)
        fg_before_grl_cst = self.grl_ins_fg_before_cons(fg_ins_feat_bf)
        fg_before_out_cst = self.fg_inshead_before(fg_before_grl_cst).sigmoid()

        da_img_loss_C, da_fg_before_loss, da_fg_before_cst = self.loss_evaluator(
            da_img=da_img_feats,  # 不做图像
            da_ins=fg_before_out,
            da_img_consist=da_img_feats_cst,
            da_ins_consist=fg_before_out_cst,
            da_ins_labels=fg_ins_labels_bf,
            targets=targets
        )

        # =============== (D) 背景 BEFORE ===============
        bg_before_grl = self.grl_ins_bg_before(bg_ins_feat_bf)
        bg_before_out = self.bg_inshead_before(bg_before_grl)
        bg_before_grl_cst = self.grl_ins_bg_before_cons(bg_ins_feat_bf)
        bg_before_out_cst = self.bg_inshead_before(bg_before_grl_cst).sigmoid()

        da_img_loss_D, da_bg_before_loss, da_bg_before_cst = self.loss_evaluator(
            da_img=da_img_feats,  # 不做图像
            da_ins=bg_before_out,
            da_img_consist=da_img_feats_cst,
            da_ins_consist=bg_before_out_cst,
            da_ins_labels=bg_ins_labels_bf,
            targets=targets
        )

        # 图像级对抗只在(A)那次里真正计算了 => 其余3次 da_img=None => da_img_loss=0
        # => final da_img_loss = da_img_loss_A

        # =============== (E) 最终合并 ===============
        losses_dict = {}
        # 1) 图像级
        final_img_loss = self.img_weight * da_img_loss_A if self.img_weight>0 else 0.0
        losses_dict["loss_da_image"] = final_img_loss

        # 2) 前景 AFTER & BEFORE
        #   AFTER => da_fg_after_loss, BEFORE => da_fg_before_loss
        #   合并 => lw_da_ins * after + (1-lw_da_ins)* before
        if self.ins_weight>0:
            fg_loss_total = self.ins_weight * (
                self.lw_da_ins * da_fg_after_loss + (1 - self.lw_da_ins)* da_fg_before_loss
            )
            losses_dict["loss_da_instance_fg"] = fg_loss_total

            bg_loss_total = self.ins_weight * (
                self.lw_da_ins * da_bg_after_loss + (1 - self.lw_da_ins)* da_bg_before_loss
            )
            bg_loss_total *= self.alpha_bg #************************这里乘了权重！！！因为背景比较多！！

            losses_dict["loss_da_instance_bg"] = bg_loss_total

        # 3) 一致性 => same approach
        if self.cst_weight>0:
            # FG
            fg_cst_total = self.cst_weight * (
                self.lw_da_ins * da_fg_after_cst + (1 - self.lw_da_ins)* da_fg_before_cst
            )
            losses_dict["loss_da_consistency_fg"] = fg_cst_total

            # BG
            bg_cst_total = self.cst_weight * (
                self.lw_da_ins * da_bg_after_cst + (1 - self.lw_da_ins)* da_bg_before_cst
            )
            # bg_cst_total *= self.alpha_bg      背景的一致性可以选择不进行！！！
            losses_dict["loss_da_consistency_bg"] = bg_cst_total


        return losses_dict
