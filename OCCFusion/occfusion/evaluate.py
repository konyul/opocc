# # Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval, SegMetric
from mmdet3d.registry import METRICS


# @METRICS.register_module()
# class EvalMetric(SegMetric):
#     """3D semantic segmentation evaluation metric.

#     Args:
#         collect_device (str, optional): Device name used for collecting
#             results from different ranks during distributed training.
#             Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
#         prefix (str): The prefix that will be added in the metric
#             names to disambiguate homonymous metrics of different evaluators.
#             If prefix is not provided in the argument, self.default_prefix
#             will be used instead. Default: None.
#         pklfile_prefix (str, optional): The prefix of pkl files, including
#             the file path and the prefix of filename, e.g., "a/b/prefix".
#             If not specified, a temp file will be created. Default: None.
#         submission_prefix (str, optional): The prefix of submission data.
#             If not specified, the submission data will not be generated.
#             Default: None.
#     """

#     def __init__(self,
#                  collect_device: str = 'cpu',
#                  prefix: Optional[str] = None,
#                  pklfile_prefix: str = None,
#                  submission_prefix: str = None,
#                  **kwargs):
#         self.pklfile_prefix = pklfile_prefix
#         self.submission_prefix = submission_prefix
#         super(SegMetric, self).__init__(
#             prefix=prefix, collect_device=collect_device)

#     def evaluation_semantic(
#         self,
#         gt_labels,
#         seg_preds,
#         label2cat,
#         ignore_index,
#         logger=None
#     ):
#         assert len(seg_preds) == len(gt_labels)
#         classes_num = len(label2cat)
#         ret_dict = dict()

#         # (클래스 개수, [TP, GT, P])를 저장할 누적 배열을 준비합니다.
#         # float 혹은 int64로 잡아두면 대용량 데이터에도 안전합니다.
#         score_accum = np.zeros((classes_num, 3), dtype=np.float64)
#         for i in range(len(gt_labels)):
#             gt_i = gt_labels[i].astype(np.int32)
#             pred_i = seg_preds[i].astype(np.int32)

#             # ignore_index는 계산에서 제외
#             mask = (gt_i != ignore_index)
#             # 클래스별로 TP, GT, P를 누적 계산
#             for j in range(classes_num):
#                 if j == 0:
#                     # 예) class 0 은 '배경이 아닌 모든 것'에 대한 geometry IoU 용도라면
#                     score_accum[j, 0] += (
#                         (gt_i[mask] != 0) & (pred_i[mask] != 0)
#                     ).sum()
#                     score_accum[j, 1] += (gt_i[mask] != 0).sum()
#                     score_accum[j, 2] += (pred_i[mask] != 0).sum()
#                 else:
#                     score_accum[j, 0] += (
#                         (gt_i[mask] == j) & (pred_i[mask] == j)
#                     ).sum()
#                     score_accum[j, 1] += (gt_i[mask] == j).sum()
#                     score_accum[j, 2] += (pred_i[mask] == j).sum()

#         # 이제 score_accum에 모든 샘플에 대한 TP/GT/P가 누적되어 있으므로,
#         # 이를 가지고 IoU = TP / (GT + P - TP)를 계산합니다.
#         mean_ious = []
#         for c in range(classes_num):
#             tp = score_accum[c, 0]
#             gt = score_accum[c, 1]
#             pred = score_accum[c, 2]
#             union = gt + pred - tp

#             if union == 0:
#                 iou = 0.0
#             else:
#                 iou = tp / union

#             mean_ious.append(iou)

#         # 각 클래스별 IoU, 그리고 class 1 이상만의 mIoU 계산
#         for i, cat_name in label2cat.items():
#             ret_dict[cat_name] = mean_ious[i]
#         # class 0은 background/geometry IoU 등으로 빼고, 1~ 나머지만 mIoU로 계산한다고 가정
#         ret_dict['mIoU'] = float(np.mean(mean_ious[1:]))

#         return ret_dict

        
#     def compute_metrics(self, results: list) -> Dict[str, float]:
#         """Compute the metrics from processed results.

#         Args:
#             results (list): The processed results of each batch.

#         Returns:
#             Dict[str, float]: The computed metrics. The keys are the names of
#             the metrics, and the values are corresponding results.
#         """
#         logger: MMLogger = MMLogger.get_current_instance()
#         if self.submission_prefix:
#             self.format_results(results)
#             return None

#         label2cat = self.dataset_meta['label2cat']
#         ignore_index = self.dataset_meta['ignore_index']

#         gt_semantic_masks = []
#         pred_semantic_masks = []
#         for eval_ann, sinlge_pred_results in results:
#             gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
#             pred_semantic_masks.append(
#                 sinlge_pred_results['pts_semantic_mask'])
#         ret_dict = self.evaluation_semantic(
#             gt_semantic_masks,
#             pred_semantic_masks,
#             label2cat,
#             ignore_index,
#             logger=logger)

#         return ret_dict
    
    
# # # single-frame gt에 해당하는 복셀 인덱스만 뽑아서 iou 계산
# # # 지금은 클래스 0,1,2에대해 그냥 다 iou 계산 

# # Copyright (c) OpenMMLab. All rights reserved.

@METRICS.register_module()
class EvalMetric(SegMetric):
    """Constant‑RAM IoU metric: class 0 IoU + mIoU over class 1+."""

    def __init__(self, collect_device: str = 'cpu', prefix: str | None = None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.score_accum: np.ndarray | None = None

    def process(self, data_batch, data_samples):
        """Accumulate TP/GT/P on‑the‑fly."""
        if self.score_accum is None:
            self.num_classes = len(self.dataset_meta['label2cat'])
            self.ignore_index = self.dataset_meta['ignore_index']
            self.score_accum = np.zeros((self.num_classes, 3), dtype=np.float64)

        for ds in data_samples:
            # ----- prediction mask -----
            if hasattr(ds, 'pred_pts_seg'):
                pred = ds.pred_pts_seg.pts_semantic_mask.astype(np.int32)
            elif 'pred_pts_seg' in ds:
                pred_dict = ds['pred_pts_seg']
                pred = pred_dict['pts_semantic_mask'].astype(np.int32) if isinstance(pred_dict, dict) else pred_dict.astype(np.int32)
            else:
                raise KeyError('prediction mask not found')

            # ----- ground‑truth mask -----
            if hasattr(ds, 'eval_ann_info'):
                gt = ds.eval_ann_info['pts_semantic_mask'].astype(np.int32)
            elif 'eval_ann_info' in ds:
                gt = ds['eval_ann_info']['pts_semantic_mask'].astype(np.int32)
            elif 'gt_pts_semantic_mask' in ds:
                gt = ds['gt_pts_semantic_mask'].astype(np.int32)
            else:
                raise KeyError('ground‑truth mask not found')

            mask = gt != self.ignore_index
            for c in range(self.num_classes):
                if c == 0:  # ★ Geometry IoU용 누적 ★
                    tp   = np.logical_and(gt != 0, pred != 0)[mask].sum()
                    gt_c = (gt != 0)[mask].sum()
                    pr_c = (pred != 0)[mask].sum()
                else:       # 기존 class 1,2
                    tp   = np.logical_and(gt == c, pred == c)[mask].sum()
                    gt_c = (gt == c)[mask].sum()
                    pr_c = (pred == c)[mask].sum()

                self.score_accum[c, 0] += tp
                self.score_accum[c, 1] += gt_c
                self.score_accum[c, 2] += pr_c

    def evaluate(self, size):
        ret, ious = {}, []
        for c, name in self.dataset_meta['label2cat'].items():
            tp, gt_c, pr_c = self.score_accum[c]
            denom = gt_c + pr_c - tp
            iou = 0.0 if denom == 0 else tp / denom

            if c == 0:
                ret['GeometryIoU'] = float(iou)   # Free IoU 대신 Geometry IoU
            else:
                ret[name] = float(iou)
                ious.append(iou)                  # mIoU용(1,2만)

        ret['mIoU'] = float(np.mean(ious)) if ious else 0.0
        return ret