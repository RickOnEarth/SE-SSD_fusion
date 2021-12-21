from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):

    def __init__(self, reader, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None,):
        super(VoxelNet, self).__init__(reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def extract_feat(self, data):
        input_features = self.reader(data["voxels"], data["num_points_per_voxel"])  # [69276, 5, 4], [69276]  -> [69276, 4]
        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        # todo: how the data change into torch datatype
        voxels = example["voxels"]                    # [69276, 5(points per voxel), 4(features per point)]
        coordinates = example["coordinates"]          # [69276, 4]
        num_points_per_voxel = example["num_points"]  # [69276], record num_points (non-zeros) in each voxel
        num_voxels = example["num_voxels"]            # [18278, 18536, 16687, 15775]
        batch_size = len(num_voxels)                  # 4
        input_shape = example["shape"][0]             # [1408, 1600,   40]

        data = dict(voxels=voxels,
                    num_points_per_voxel=num_points_per_voxel,
                    coors=coordinates,
                    batch_size=batch_size,
                    input_shape=input_shape,)

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        # for debug, self.bbox_head.predict(example, preds, self.test_cfg)
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
