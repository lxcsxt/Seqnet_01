import torch
from torch import nn
from torch.nn import functional as F


def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):
    """
    Consistency regularization as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \\sum_{i,j}||\frac{1}{|I|}\\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    """
    # 如果 instance 数为0，直接返回0
    if ins_fea is None or ins_fea.size(0) == 0:
        return ins_fea.new_zeros(1, dtype=torch.float32)

    loss = []
    len_ins = ins_fea.size(0)
    # intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
    for img_fea_per_level in img_feas:
        N, A, H, W = img_fea_per_level.shape
        img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
        img_feas_per_level = []
        # assert N==2, \
        #     "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
        for i in range(N):
            # img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
            img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(len_ins // N, 1)
            img_feas_per_level.append(img_fea_mean)
        if len_ins % N != 0:
            img_feas_per_level.append(img_fea_per_level[N - 1].view(1, 1).repeat(len_ins % N, 1))
        img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
        loss_per_level = torch.abs(img_feas_per_level - ins_fea)
        loss.append(loss_per_level)
    loss = torch.cat(loss, dim=1)
    if size_average:
        return loss.mean()
    return loss.sum()




class DALossComputation:
    """
    This class computes the DA loss.
    """

    def __init__(self):
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image["domain_labels"]
            mask_per_image = (
                is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1, dtype=torch.bool)
            )
            masks.append(mask_per_image)
        return masks

    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            da_img (list[Tensor] or []): image-level logits
            da_img_consist (list[Tensor] or []): image-level consistency
            da_ins (Tensor or None): instance-level logits
            da_ins_consist (Tensor or None): instance-level consistency
            da_ins_labels (Tensor): instance-level domain labels
            targets (list[Dict]): each dict has "domain_labels", used for image-level mask

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        # ----------------- 0) Init to zero ---------------
        da_img_loss = 0.0
        da_ins_loss = 0.0
        da_consist_loss = 0.0

        # ----------------- 1) image-level domain alignment ---------------
        # 如果 da_img 是空列表或 None，跳过 flatten
        if da_img is not None and len(da_img) > 0:
            # 准备mask
            masks = self.prepare_masks(targets)
            masks = torch.cat(masks, dim=0)

            da_img_flattened = []
            da_img_labels_flattened = []

            for da_img_per_level in da_img:
                # da_img_per_level shape: [N, 1, H, W] or [N, A, H, W]
                N, A, H, W = da_img_per_level.shape
                da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)  # => [N, H, W, A]
                da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
                # domain_labels=1 => source, => set label=1
                da_img_label_per_level[masks, :] = 1

                da_img_per_level = da_img_per_level.reshape(N, -1)
                da_img_label_per_level = da_img_label_per_level.reshape(N, -1)

                da_img_flattened.append(da_img_per_level)
                da_img_labels_flattened.append(da_img_label_per_level)

            if len(da_img_flattened) > 0:
                da_img_flattened = torch.cat(da_img_flattened, dim=0)
                da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
                da_img_loss = F.binary_cross_entropy_with_logits(da_img_flattened, da_img_labels_flattened)
            else:
                da_img_loss = 0.0
        else:
            # da_img is empty => skip
            da_img_loss = 0.0

        # ----------------- 2) instance-level domain alignment ---------------
        # 如果 da_ins 为空, 表示此分支不做实例级对抗
        if da_ins is not None and da_ins.numel() > 0:
            da_ins_loss = F.binary_cross_entropy_with_logits(
                da_ins.view(-1), da_ins_labels.float().view(-1)
            )
        else:
            da_ins_loss = 0.0




        print("da_img_consist:", da_img_consist)
        print("da_ins_consist:", da_ins_consist)


        # ----------------- 3) consistency loss ---------------
        # 如果 da_img_consist 或 da_ins_consist 是 None/空 => skip
        can_do_consistency = (da_img_consist is not None 
                            and len(da_img_consist) > 0  # 列表非空
                            and da_ins_consist is not None 
                            and da_ins_consist.numel() > 0 )
        if can_do_consistency:
            da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)
        else:
            da_consist_loss = 0.0

        return da_img_loss, da_ins_loss, da_consist_loss
