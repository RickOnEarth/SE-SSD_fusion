import itertools
import logging
from pathlib import Path

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

data_root_prefix = "/mengxing/Data/Sets"

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [dict(num_class=1, class_names=["Car"],),]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
box_coder = dict(type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,)

# torch.set_printoptions(precision=4, sci_mode=False)
my_paras = dict(
    batch_size=1,   #4
    data_mode="train",        # "train" or "trainval": the set to train the model;
    enable_ssl=True,         # Ensure "False" in CIA-SSD training
    eval_training_set=False,  # True: eval on "data_mode" set; False: eval on validation set.[Ensure "False" in training; Switch in Testing]

    # unused
    enable_difficulty_level=False,
    remove_difficulty_points=False,  # act with neccessary condition: enable_difficulty_level=True.
    gt_random_drop=-1,
    data_aug_random_drop=-1,
    far_points_first=False,
    data_aug_with_context=-1,        # enlarged size for w and l in data aug.
    gt_aug_with_context=-1,
    gt_aug_similar_type=False,
    min_points_in_gt=-1,
    loss_iou=None,
)

# model settings
model = dict(
    type="PointPillars_FUSION_MX",
    #type="VoxelNet",
    pretrained=None,
    reader=dict(type="PillarFeatureNet",
                num_filters=[64],
                with_distance=False,
                norm_cfg=norm_cfg,
                ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=norm_cfg,),

    neck=dict(
        type="RPN",
        #type="SSFA_MX",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),

    bbox_head=dict(
        type="MultiGroupHead_MX_PP",
        mode="3d",
        in_channels=sum([128, 128, 128]),          #in_channels=sum([128,]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], codewise=True, loss_weight=2.0, ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(type="WeightedSoftmaxClassificationLoss", name="direction_classifier", loss_weight=0.2,),
        direction_offset=0.0,
        #loss_iou=my_paras['loss_iou'],
    ),
)

target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.6, 3.9, 1.56],  # w, l, h
            anchor_ranges=[0, -40.0, -1.0, 70.4, 40.0, -1.0],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="Car",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=8,
    debug=False,
    enable_similar_type=True,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.5,         #nms_iou_threshold=0.01,
    ),
    score_threshold=0.3,               #0.05
    post_center_limit_range=[0, -40.0, -5.0, 70.4, 40.0, 5.0],
    max_per_img=100,
)

# dataset settings
dataset_type = "KittiDataset"

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[0, -39.68, -3, 69.12, 39.68, 1],
    voxel_size=[0.16, 0.16, 4.0],
    max_points_in_voxel=100,
    max_voxel_num=12000,
    far_points_first=my_paras['far_points_first'],           #???
)

test_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget_MX_PP", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

data_root = data_root_prefix + "/kitti_se-ssd"

test_anno = data_root+"/kitti_infos_train.pkl"

data = dict(
    samples_per_gpu=1,  # batch_size: 4
    workers_per_gpu=2,  # default: 2
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# for cia optimizer
optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,)  # learning policy in training hooks



checkpoint_config = dict(interval=1)
log_config = dict(interval=10,hooks=[dict(type="TextLoggerHook"),],) # dict(type='TensorboardLoggerHook')

# runtime settings
total_epochs = 60
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"

TAG = 'pretrained_model'   #'pretrained_model'            #'work_dir_debug'
work_dir = "/mengxing/LiDAR_Detection/SE-SSD/model_dir/pointpillars/" + TAG

load_from = "/mengxing/LiDAR_Detection/SE-SSD/model_dir/pointpillars/pretrained_model/epoch_60.pth"         #for training
#load_from = None
resume_from = None
workflow = [("train", 60), ("val", 1)] if my_paras['enable_ssl'] else [("train", 60), ("val", 1)]
save_file = False if TAG == "debug" or TAG == "exp_debug" or Path(work_dir, "Det3D").is_dir() else True

fusion_cfg = dict(
    optimizer=dict(
        optimizer_type='adam_optimizer',
        adam_optimizer=dict(
            learning_rate=dict(
                lr_type='one_cycle',
                one_cycle=dict(
                    lr_max=0.003,
                    moms=[0.95, 0.85],
                    div_factor=10.0,  # original 10
                    pct_start=0.4,
                )
            ),
            weight_decay=0.01,
            amsgrad=3,                          #???
        ),
        fixed_weight_decay=True,
        use_moving_average=False
    ),
    steps=37120*3,  # 112215 #113715 #111360 # 619 * 50, super converge. increase this to achieve slightly better results original 30950
    steps_per_eval=3712,  # 7481 # 619 * 5
    save_checkpoints_secs=1800,  # half hour 1800
    save_summary_steps=10,
    enable_mixed_precision=False,  # for fp16 training, don't use this.
    loss_scale_factor=512.0,
    clear_metrics_every_epoch=True,
    checkpoint_saved_dir='/mengxing/LiDAR_Detection/SE-SSD/model_dir/fusion_pointpillars/work_dir'
    #detection_2d_path: "../d2_detection_data"
)

