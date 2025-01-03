from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from models.oim import OIMLoss
from models.resnet import build_resnet

import torch
import torch.nn as nn


import torch.nn.functional as F

class ScaledFeatureFusionModule(nn.Module):
    def __init__(self, in_channels_res2=256, in_channels_res3=512, in_channels_res4=1024, out_channels=1024):
        super(ScaledFeatureFusionModule, self).__init__()

        # 将每个特征图的通道数统一为 out_channels（1024）
        self.conv_res2 = nn.Conv2d(in_channels_res2, out_channels, kernel_size=1)
        self.conv_res3 = nn.Conv2d(in_channels_res3, out_channels, kernel_size=1)
        self.conv_res4 = nn.Conv2d(in_channels_res4, out_channels, kernel_size=1)

        # 定义全连接层来学习不同尺度的权重
        self.fc_res2 = nn.Linear(out_channels, 1)
        self.fc_res3 = nn.Linear(out_channels, 1)
        self.fc_res4 = nn.Linear(out_channels, 1)

        # 输出层，输出加权后的特征通道数
        self.fc_out = nn.Linear(3, 3)

        # 可选：对于最终融合后的特征，我们可能希望加上一个卷积层来进一步处理
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 添加一个1x1卷积来确保融合后的通道数为1024
        self.conv_final = nn.Conv2d(out_channels, 1024, kernel_size=1)

    def forward(self, feat_res2, feat_res3, feat_res4):
        """
        feat_res2, feat_res3, feat_res4: 分别是不同尺度的特征图 (B, C, H, W)
        """
        # 使用 1x1 卷积将特征图的通道数统一
        feat_res2 = self.conv_res2(feat_res2)  # (B, out_channels, H, W)
        feat_res3 = self.conv_res3(feat_res3)
        feat_res4 = self.conv_res4(feat_res4)

        # 计算每一层特征图的统计量（这里简单地使用平均值作为统计量）
        weight_res2 = torch.mean(feat_res2, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        weight_res3 = torch.mean(feat_res3, dim=[2, 3], keepdim=True)
        weight_res4 = torch.mean(feat_res4, dim=[2, 3], keepdim=True)

        # 通过全连接层获得每个特征图的权重系数
        weight_res2 = self.fc_res2(weight_res2.view(weight_res2.size(0), -1))  # (B, 1)
        weight_res3 = self.fc_res3(weight_res3.view(weight_res3.size(0), -1))  # (B, 1)
        weight_res4 = self.fc_res4(weight_res4.view(weight_res4.size(0), -1))  # (B, 1)

        # 结合各个尺度的权重
        weights = torch.cat([weight_res2, weight_res3, weight_res4], dim=1)  # (B, 3)
        scale_weights = torch.sigmoid(self.fc_out(weights))  # (B, 3)

        # 对所有特征图进行上采样，使得它们的空间尺寸一致
        target_size = feat_res4.size()[2:]  # (H, W) of feat_res4
        feat_res2 = F.interpolate(feat_res2, size=target_size, mode='bilinear', align_corners=False)
        feat_res3 = F.interpolate(feat_res3, size=target_size, mode='bilinear', align_corners=False)

        # 进行加权融合
        fused_features = feat_res2 * scale_weights[:, 0].view(-1, 1, 1, 1) + \
                         feat_res3 * scale_weights[:, 1].view(-1, 1, 1, 1) + \
                         feat_res4 * scale_weights[:, 2].view(-1, 1, 1, 1)

        # 可选的卷积层来进一步处理融合后的特征
        fused_features = self.conv_out(fused_features)

        # 最后使用1x1卷积将通道数减少到1024，适应后续操作
        fused_features = self.conv_final(fused_features)

        return OrderedDict([["fused_features", fused_features]])







class SeqNet(nn.Module):
    def __init__(self, cfg):#它接收一个配置参数 cfg，这个配置参数通常包含了模型各个部分的超参数设置等信息
        super(SeqNet, self).__init__()
        # box_head是用于目标检测的头部模块。
        backbone, box_head = build_resnet(name="resnet50", pretrained=True)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],#同时指定了每个位置的锚点数量，这个数量由之前创建的 anchor_generator 来确定。
        )

        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )

        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,

            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )



        #Fast RCNN Predictor (目标分类和边框回归)输入特征维度为 2048，输出类别数为 2，可推测这里是针对二分类任务进行设置。
        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = deepcopy(box_head)#reid_head 可能和行人重识别等相关任务中的特征提取有关。

        from torchvision.ops import MultiScaleRoIAlign
        #用于将不同尺度的感兴趣区域特征转换为固定尺寸的特征表示。********怎么实现的？？？？？？*******
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["fused_features"], output_size=14, sampling_ratio=2
        )

        # 用于预测目标的边界框位置。
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)

        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,#
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # Feature fusion module (add this part)
        self.feature_fusion = ScaledFeatureFusionModule(
            in_channels_res2=256,  # 根据实际通道数设置
            in_channels_res3=512,
            in_channels_res4=1024,
            out_channels=1024
        )


        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID


    #query_img_as_gallery：布尔值，指示是否将查询图像作为gallery（图像库）来处理。
    #若为 True，则认为查询图像中所有的人都是待检测的人，并且目标框（gt box）应该是检测框中第一个。
    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]

        #*****************   这里我认为是可以修改的！transform  *************************
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        # 使用特征融合模块融合不同尺度的特征
        feat_res2 = features["feat_res2"]
        feat_res3 = features["feat_res3"]
        feat_res4 = features["feat_res4"]

        # 调用特征融合模块
        fused_features = self.feature_fusion(feat_res2, feat_res3, feat_res4)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(fused_features , boxes, images.image_sizes)
            # *****************   这里我认为是可以修改的！reid_head[太单薄了！] *************************
            box_features = self.roi_heads.reid_head(box_features)
            # *****************   我认为可以将行人的特征进行增强，然后再去转换成行人嵌入特征   *************************
            embeddings, _ = self.roi_heads.embedding_head(box_features)#提取行人嵌入特征
            return embeddings.split(1, 0)#最后将得到的嵌入特征 embeddings 按照维度 0（通常对应批量维度，例如如果是对多个目标进行处理，就会沿着每个目标对应的维度拆分）拆分成单个元素的列表返回，这样返回的格式可能更便于后续针对每个目标进行单独的处理、分析或者与其他相关特征进行匹配等操作。

        else:
            # gallery
            # 通过区域提议网络生成一系列的候选区域提议
            proposals, _ = self.rpn(images, fused_features , targets)
            detections, _ = self.roi_heads(
                fused_features , proposals, images.image_sizes, targets, query_img_as_gallery
            )
            #将得到的检测结果 detections 以及当前图像的尺寸 images.image_sizes 和之前记录的原始图像尺寸 original_image_sizes 作为参数传入，对检测结果进行后处理，
            #例如将检测框的坐标等信息从处理后的图像尺寸空间转换回原始图像尺寸对应的坐标空间，使其符合实际应用中对检测结果在原始图像上展示等需求。
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        # 使用特征融合模块融合不同尺度的特征
        feat_res2 = features["feat_res2"]
        feat_res3 = features["feat_res3"]
        feat_res4 = features["feat_res4"]

        # 调用特征融合模块
        fused_features = self.feature_fusion(feat_res2, feat_res3, feat_res4)

        proposals, proposal_losses = self.rpn(images, fused_features, targets)
        #对每个提议区域进行目标分类和边界框回归等，同时计算在这个过程中产生的损失（detector_losses），这里使用 _ 忽略了第一个返回值（可能是经过 roi_heads 处理后的一些中间结果等
        _, detector_losses = self.roi_heads(fused_features, proposals, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses




class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,#是用于目标分类和边框回归的预测器对象（之前在外部初始化好传入的，比如 FastRCNNPredictor 类型，用于基于感兴趣区域进行目标分类和边框位置预测）
        reid_head,#reid_head 可能和行人重识别等相关任务中的特征提取有关（也是外部传入已初始化的对象）
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding()#从名字推测它是用于生成具有某种归一化感知的嵌入特征的模块，这个嵌入特征可能在后续的行人重识别等任务中用于衡量不同图像中行人是否为同一人等情况
        self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        self.postprocess_proposals = self.postprocess_detections

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor]) - 加权融合后的特征
            proposals (List[Tensor[N, 4]]) - 每个图片的候选框
            image_shapes (List[Tuple[H, W]]) - 每个图像的尺寸
            targets (List[Dict]) - 每个图片的目标信息，包含GT框、标签等
            query_img_as_gallery (bool) - 是否将查询图像当作图库进行处理
        """

        #这是一个条件判断语句，当模型处于训练模式（通过 self.training 属性判断，继承自 nn.Module 的相关机制）时，
        # 调用 self.select_training_samples 方法（这个方法应该是在父类或者自身类中定义的用于从候选框和目标信息中选择合适的训练样本的函数
        # ，具体实现细节可能涉及根据一定规则筛选样本、分配标签以及确定边界框回归的目标等操作），传入候选框 proposals 和目标信息 targets，
        # 返回经过筛选和处理后的候选框 proposals、以及新生成的提议区域的行人身份标签（proposal_pid_labels）和边界框回归目标（proposal_reg_targets），
        # 这里使用 _ 忽略了第二个返回值（可能是该方法返回的其他中间信息，但从当前代码逻辑看暂时不需要使用它），这些返回值将用于后续的损失计算以及模型训练相关操作。
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )

        # ------------------- Faster R-CNN head ------------------ #
        # 使用加权融合后的特征来提取 proposal 特征
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        #传入经过前面处理后的特征中的 feat_res5（可能是特定的某个特征层输出，根据具体网络结构和特征提取逻辑而定），
        #得到目标分类得分（proposal_cls_scores）和边界框回归结果（proposal_regs），
        #分类得分用于判断每个候选框对应的物体属于各类别的概率，边界框回归结果用于调整候选框的位置和大小，使其更接近真实目标框。
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )

        if self.training:
            #计算最终的预测框坐标信息，将边界框回归的结果应用到原始候选框上得到实际的预测框位置），得到预测框信息 boxes，每个元素对应一张图像中的预测框集合。
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            #通过列表推导式对每个图像中的预测框张量进行 detach 操作，这一步主要是为了将这些预测框张量从计算图中分离出来，避免在后续一些不需要梯度计算的操作中产生不必要的梯度传播等情况，同时保留其数值用于后续处理。
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            #这次传入预测框 boxes 和目标信息 targets，重新筛选和处理预测框，得到新的预测框 boxes、预测框对应的行人身份标签（box_pid_labels）以及边界框回归目标（box_reg_targets），同样使用 _ 忽略第二个返回值，这些新的信息将用于后续更准确的损失计算等训练相关操作。
            boxes, _, box_pid_labels, box_reg_targets = self.select_training_samples(boxes, targets)
        else:
            # 使用继承自父类的 postprocess 方法来处理 proposals
            #通过这个后处理方法对候选框相关信息进行处理，得到最终的预测框 boxes、
            #得到对应的得分 scores（用于衡量预测框包含目标的置信度等情况）
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # 当将查询图像当作图库时，强制包括GT框，并禁用CWS
            cws = False
            gt_box = [targets[0]["boxes"]]
            #调用 self.box_roi_pool 对融合后的特征 features、真实目标框 gt_box 和图像尺寸 image_shapes
            # 进行感兴趣区域池化操作，得到真实目标框对应的特征 gt_box_features。
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head(gt_box_features)
            embeddings, _ = self.embedding_head(gt_box_features)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        #如果没有预测到任何检测框，直接返回GT框和嵌入
        #这是一个边界情况处理逻辑，当预测得到的检测框列表中第一个元素（对应第一张图像的检测框张量）的形状在维度 0（通常是表示框的数量维度）上为 0，即没有预测到任何检测框时，进行如下操作：
        #首先通过 assert 断言当前处于非训练模式，因为在训练模式下通常不会出现这种直接返回的情况（训练过程更关注损失计算等基于有预测结果的情况）。
        #然后根据是否有之前计算得到的真实目标框信息（gt_det 是否存在）来设置 boxes、labels、scores 和 embeddings 的值，如果有 gt_det，则使用其中的真实目标框信息作为 boxes，设置 labels 和 scores 为默认值（例如都设为 1，可能表示默认将真实框当作有目标且置信度为 1 的情况，具体含义取决于应用场景），使用 gt_det 中的嵌入特征作为 embeddings；如果没有 gt_det，则将 boxes、labels、scores 和 embeddings 都初始化为对应维度的全 0 张量，分别表示没有检测框、没有标签、没有得分以及没有嵌入特征的情况。
        #最后返回一个包含字典的列表（字典中包含 boxes、labels、scores 和 embeddings 信息，模拟正常的检测结果格式）以及一个空列表（可能是对应损失信息的位置，但此时没有损失需要返回，因为没有实际的预测结果参与训练相关操作）。
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, 256)
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- Baseline head -------------------- #
        # 使用加权融合后的特征提取最终的框特征
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)
        box_regs = self.box_predictor(box_features["feat_res5"])
        box_embeddings, box_cls_scores = self.embedding_head(box_features)
        if box_cls_scores.dim() == 0:
            box_cls_scores = box_cls_scores.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )
            loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels)
            losses.update(loss_box_reid=loss_box_reid)
        else:
            # 使用更高的 NMS 阈值处理预测框
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                    )
                )
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        从 proposals 中获取预测框。
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # 移除背景类别的预测框
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        处理检测框，支持 First Classification Score (FCS) 和 Confidence Weighted Similarity (CWS)
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # 使用 Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # 将框、分数和嵌入分解到每个图像
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.ones(scores.size(0), device=device)

            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels






class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )
