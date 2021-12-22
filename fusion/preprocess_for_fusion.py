import torch
import numpy as np
from det3d.core.bbox import box_torch_ops
from det3d.core.bbox import box_np_ops
from fusion.d2_reader import detection_2d_reader

def preprocess_for_fusion(model, example, all_preds_dicts):
    #注意：实际batch_size应该保持为1
    # 单分类，config中len(tasks)=1
    task_id = 0
    preds_dict = all_preds_dicts[task_id]

    batch_anchors = example["anchors"]
    batch_size = batch_anchors[task_id].shape[0]

    assert model.bbox_head.encode_background_as_zeros is True
    num_class_with_bg = model.bbox_head.num_classes[task_id]

    batch_cls_preds = preds_dict["cls_preds"].view(batch_size, -1, num_class_with_bg)  # [bs=1, 70400, 1]
    batch_box_preds = preds_dict["box_preds"].view(batch_size, -1, model.bbox_head.box_n_dim)  # [bs=1, 70400, 7]
    batch_task_anchors = example["anchors"][task_id].view(batch_size, -1, model.bbox_head.box_n_dim)   # [bs=1, 70400, 7]

    batch_anchors_mask = [None] * batch_size
    if "anchors_mask" in example:       #not in example
        pass    #TODO for pp

    batch_rect = example['calib']['rect']          #torch.Size([1, 4, 4])
    batch_Trv2c = example['calib']['Trv2c']          #torch.Size([1, 4, 4])
    batch_P2 = example['calib']['P2']
    batch_metadata = [metadata for metadata in example['metadata']]

    for box_preds, cls_preds, rect, Trv2c, P2, metadata, a_mask in zip(
        batch_box_preds, batch_cls_preds, batch_rect, batch_Trv2c, batch_P2, batch_metadata, batch_anchors_mask):

        if a_mask is not None:      #is None
            # box_preds = box_preds[a_mask]
            # cls_preds = cls_preds[a_mask]
            pass    #TODO for pp

        #TODO: score mask to filter 3d candidates

        # box_preds = box_preds.float()                           #shape: [70400, 7]
        # cls_preds = cls_preds.float()                           #shape: [70400, 1]
        rect = rect.float()                                     #shape: [4, 4]
        Trv2c = Trv2c.float()                                   #shape: [4, 4]
        P2 = P2.float()

        total_scores = torch.sigmoid(cls_preds)
        dis_to_lidar = torch.norm(box_preds[:, :2], p=2, dim=1, keepdim=True) / 82.0

        final_box_preds = box_preds
        final_scores = total_scores

        final_box_preds_camera = box_torch_ops.box_lidar_to_camera(final_box_preds, rect, Trv2c)

        ###project bboxes to image###
        locs = final_box_preds_camera[:, :3]
        dims = final_box_preds_camera[:, 3:6]
        angles = final_box_preds_camera[:, 6]
        camera_box_origin = [0.5, 1.0, 0.5]
        box_corners = box_torch_ops.center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=1)
        box_corners_in_image = box_torch_ops.project_to_image(box_corners, P2)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        img_height = metadata['image_shape'][0]
        img_width = metadata['image_shape'][1]

        minxy[:, 0] = torch.clamp(minxy[:, 0], min=0, max=img_width)
        minxy[:, 1] = torch.clamp(minxy[:, 1], min=0, max=img_height)
        maxxy[:, 0] = torch.clamp(maxxy[:, 0], min=0, max=img_width)
        maxxy[:, 1] = torch.clamp(maxxy[:, 1], min=0, max=img_height)

        box_2d_preds = torch.cat([minxy, maxxy], dim=1)

        processed_preds_dict = {
            "bbox_2d_projected": box_2d_preds,                           #torch.Size([70400, 4])
            "box3d_camera": final_box_preds_camera,
            "box3d_lidar": final_box_preds,
            "scores": final_scores,
            # "label_preds": label_preds,
            "image_idx": metadata['image_idx'],
        }

        top_2d_predictions = detection_2d_reader(metadata['image_idx'])
        if len(top_2d_predictions > 200):                              #注意top_2d_predictions已经按score大小排好序了
            top_2d_predictions = top_2d_predictions[:200, :]
        box_2d_detector = top_2d_predictions[:, :4]
        box_2d_scores = top_2d_predictions[:, 4].reshape(-1, 1)

        num_combination = 10000000                                          #这里按实际需要设值大小
        overlaps1 = np.zeros((num_combination, 4), dtype=np.float32)
        tensor_index1 = np.zeros((num_combination, 2), dtype=np.int64)
        overlaps1[:, :] = -1                                        #不需要？？
        tensor_index1[:, :] = -1

        iou_test, tensor_index, max_num = box_np_ops.build_stage2_training(box_2d_preds.detach().cpu().numpy(),
                                                                   box_2d_detector,
                                                                   -1,
                                                                   final_scores.detach().cpu().numpy(),
                                                                   box_2d_scores,
                                                                   dis_to_lidar.detach().cpu().numpy(),
                                                                   overlaps1,
                                                                   tensor_index1)
        """
        iou_test.shape: (10000000, 4). 4分别是[iou, score_3d, score_2d, dis_to_lidar]
        tensor_index.shape: (10000000, 2). 2分别是[2d pred idx, 3d pred idx]
        max_num: 实际排列组合有效的数目（每个3d condidate至少占一个）
        """
        iou_test_tensor = torch.FloatTensor(iou_test)
        tensor_index_tensor = torch.LongTensor(tensor_index)
        iou_test_tensor = iou_test_tensor.permute(1, 0)
        iou_test_tensor = iou_test_tensor.reshape(1, 4, 1, num_combination)     #torch.Size([1, 4, 1, 10000000])
        tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)                #torch.Size([10000000, 2])

        if max_num == 0:
            non_empty_iou_test_tensor = torch.zeros(1, 4, 1, 2)
            non_empty_iou_test_tensor[:, :, :, :] = -1
            non_empty_tensor_index_tensor = torch.zeros(2, 2)
            non_empty_tensor_index_tensor[:, :] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:, :, :, :max_num]       #torch.Size([1, 4, 1, max_num])
            non_empty_tensor_index_tensor = tensor_index_tensor[:max_num, :]     #torch.Size([max_num, 2])

        return processed_preds_dict, top_2d_predictions, non_empty_iou_test_tensor, non_empty_tensor_index_tensor





def preprocess_for_fusion_bak(model, example, all_preds_dicts):
    #注意：下面的代码都是建立在batch_size为1的前提下！！
    # 单分类，config中len(tasks)=1
    task_id = 0

    box_preds = all_preds_dicts[task_id]['box_preds']
    cls_preds = all_preds_dicts[task_id]['cls_preds']
    box_code_size = model.bbox_head.box_coder.n_dim    #7
    box_preds = box_preds.view(-1, box_code_size)      #torch.Size([70400, 7])

    num_class_with_bg = model.bbox_head.num_classes[task_id]  # 1

    cls_preds = cls_preds.view(-1, num_class_with_bg)

    anchors = example['anchors'][task_id].view(-1, box_code_size)

    #decode box_preds
    box_preds = model.bbox_head.box_coder.decode_torch(box_preds, anchors)      #torch.Size([70400, 7])

    rect = example['calib']['rect'].squeeze(0)          #torch.Size([4, 4])
    Trv2c = example['calib']['Trv2c'].squeeze(0)          #torch.Size([4, 4])
    P2 = example['calib']['P2'].squeeze(0)



    import pdb
    pdb.set_trace()




    return