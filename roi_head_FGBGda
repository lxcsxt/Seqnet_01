import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops


class SeqRoIHeadsDa(RoIHeads):
    def __init__(self, faster_rcnn_predictor, reid_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding()
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

        self.memory = None

    def select_training_samples_gt(self, proposals, targets, is_source=True):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, matched_gt_boxes

    """
    输入参数
        proposals: 每张图像的候选框列表，每个元素是形状为 [N, 4] 的张量（N为候选框数量）。

        gt_boxes: 每张图像的真实框列表，每个元素是形状为 [M, 4] 的张量（M为真实框数量）。

        gt_labels: 每张图像的真实标签列表，每个元素是形状为 [M] 的张量。

    输出
        matched_idxs: 每张图像中每个候选框匹配的真实框索引（若未匹配则为负值或0）。

        labels: 每个候选框的类别标签（正样本、负样本或忽略）
    """
    def assign_targets_to_proposalsFGBG(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        #-------------------------------------------------------------
        #                   暂时调整阈值
        #-------------------------------------------------------------
        self.proposal_matcher.high_threshold=0.25
        self.proposal_matcher.low_threshold=0.1      

        #遍历每张图像的候选框、真实框和标签。
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):


            #3. 处理无真实框的图像（背景图）
            #所有候选框的匹配索引设为 0（实际无意义，后续标签会覆盖为背景）。
            # 所有候选框标签设为 0（背景类）。
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:#处理含真实框的图像
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算IoU矩阵：形状 [num_gt, num_proposals]
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

                
                # ============== 新增：统一多类为前景=1 ==============
                # 这一步会把 labels_in_image 中所有 >=1 的值统统变成1，
                # 背景=0 保持0，-1 维持不变。
                valid_mask = (labels_in_image >= 1)
                labels_in_image[valid_mask] = 1
                # ==================================================

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)


        #-------------------------------------------------------------
        #                   恢复阈值
        #-------------------------------------------------------------
        self.proposal_matcher.high_threshold=0.5
        self.proposal_matcher.low_threshold=0.5


        return matched_idxs, labels
    # boxes update
    def select_training_samples_da(self, proposals, targets):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        domain_labels = [t["domain_labels"] for t in targets]

        # get matching gt indices for each proposal
        # 在 assign_targets_to_proposals 里，我们已经把多类都合并成1 / 0 / -1
        matched_idxs, labels = self.assign_targets_to_proposalsFGBG(proposals, gt_boxes, gt_labels)

        # for is_source, labels_per_image in zip(domain_labels, labels):
        #     labels_per_image[:] = 0
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []

        all_domains = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            # print(proposals[img_id].shape)

            domain_label = (
                torch.ones_like(labels[img_id], dtype=torch.uint8)
                if domain_labels[img_id].any()
                else torch.zeros_like(labels[img_id], dtype=torch.uint8)
            )
            all_domains.append(domain_label)

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, all_domains, matched_gt_boxes

    #----------------------------------------------------------------------------
    #
    #       拆分前景  背景函数
    #
    #----------------------------------------------------------------------------
    def split_fg_bg_embeddings(self, embeddings, labels_list, domain_list):
        """
        将embedding根据labels(0=背景,1=前景,-1=ignore)拆分成:
        fg_embeds, fg_domains, bg_embeds, bg_domains
        参数:
        embeddings: [sumN, D]  (对多张图像合并后的ROI特征)
        labels_list: list[Tensor], 每张图的labels => 0/1/-1
        domain_list: list[Tensor], 每张图的domain => 0/1 (源/目标)
        返回:
        (fg_embeds, fg_domains, bg_embeds, bg_domains)
        """
        # 先算出每张图像proposal数
        lens = [len(lbl) for lbl in labels_list]
        splitted_emb = embeddings.split(lens, dim=0)

        all_fg_embeds = []
        all_fg_domains = []
        all_bg_embeds = []
        all_bg_domains = []

        for lbls, doms, emb in zip(labels_list, domain_list, splitted_emb):
            fg_mask = (lbls == 1)
            bg_mask = (lbls == 0)

            if fg_mask.any():
                all_fg_embeds.append(emb[fg_mask])
                all_fg_domains.append(doms[fg_mask])
            if bg_mask.any():
                all_bg_embeds.append(emb[bg_mask])
                all_bg_domains.append(doms[bg_mask])

        if len(all_fg_embeds)>0:
            fg_embeds  = torch.cat(all_fg_embeds, dim=0)
            fg_domains = torch.cat(all_fg_domains, dim=0)
        else:
            fg_embeds  = torch.zeros(0, embeddings.size(-1), device=embeddings.device)
            fg_domains = torch.zeros(0, dtype=torch.uint8, device=embeddings.device)

        if len(all_bg_embeds)>0:
            bg_embeds  = torch.cat(all_bg_embeds, dim=0)
            bg_domains = torch.cat(all_bg_domains, dim=0)
        else:
            bg_embeds  = torch.zeros(0, embeddings.size(-1), device=embeddings.device)
            bg_domains = torch.zeros(0, dtype=torch.uint8, device=embeddings.device)

        return (fg_embeds, fg_domains, bg_embeds, bg_domains)



    def extract_da(
        self,
        features,
        proposals,
        image_shapes,
        targets=None,
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if self.training:
            (
                proposals,
                matched_idxs,
                labels_before,
                regression_targets,
                domain_labels_before,
                _,
            ) = self.select_training_samples_da(proposals, targets)

        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(proposal_features["feat_res5"])
        
        # 拆分前景/背景 => (fg_embeds_before, fg_domains_before, bg_embeds_before, bg_domains_before)
        if self.training and len(labels_before) > 0:
            (fg_embeds_before, fg_domains_before,
            bg_embeds_before, bg_domains_before) = self.split_fg_bg_embeddings(
                proposal_features["feat_res5"], labels_before, domain_labels_before
            )
        
        
    
        boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
        boxes = [boxes_per_image.detach() for boxes_per_image in boxes]    
        boxes, matched_idxs, labels, regression_targets, domain_labels, _ = self.select_training_samples_da(
            boxes, targets
        )
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)
        box_embeddings, box_cls_scores = self.embedding_head(box_features)


        if self.training and len(labels) > 0:
            (fg_embeds, fg_domains,
            bg_embeds, bg_domains) = self.split_fg_bg_embeddings(
                box_embeddings, labels, domain_labels
            )
        return (fg_embeds, fg_domains,
            bg_embeds, bg_domains,
            fg_embeds_before, fg_domains_before,
            bg_embeds_before, bg_domains_before)
    


    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False, is_source=True):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(proposals, targets)

        # ------------------- Faster R-CNN head ------------------ #

        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(proposal_features["feat_res5"])

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, matched_idxs, box_pid_labels, box_reg_targets, matched_gt_boxes = self.select_training_samples_gt(
                boxes, targets
            )

        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_proposals(proposal_cls_scores, proposal_regs, proposals, image_shapes)

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head(gt_box_features, is_source)
            embeddings, _ = self.embedding_head(gt_box_features)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, 256)
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- Baseline head -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features, is_source)
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
            if is_source:
                loss_box_reid_s = self.memory(box_embeddings, box_pid_labels, is_source=True)
                losses.update(loss_box_reid_s=loss_box_reid_s)
            else:
                loss_box_reid_t = self.memory(box_embeddings, box_pid_labels)
                losses.update(loss_box_reid_t=loss_box_reid_t)
                if not math.isfinite(loss_box_reid_t):
                    pass
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
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
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(dict(boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]))
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
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
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
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
        super().__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim
        self.temp = None
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
            self.temp = embeddings
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
