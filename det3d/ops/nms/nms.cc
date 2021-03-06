#include "nms.h"
#include "nms_cpu.h"
PYBIND11_MODULE(nms, m)
{
    m.doc() = "non_max_suppression asd";
    m.def("non_max_suppression", &non_max_suppression<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression", &non_max_suppression<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    //m.def("IOU_weighted_rotate_non_max_suppression_cpu", &IOU_weighted_rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou",
    //      "box"_a = 1, "box_corners"_a = 2,  "standup_iou"_a = 3, "thresh"_a = 4, "scores"_a = 5, "sigma"_a = 6, "labels"_a = 7, "dirs"_a = 8);
    //m.def("IOU_weighted_rotate_non_max_suppression_cpu", &IOU_weighted_rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou",
    //      "box"_a = 1, "box_corners"_a = 2,  "standup_iou"_a =3, "thresh"_a = 4, "scores"_a = 5, "sigma"_a = 6, "labels"_a = 7, "dirs"_a = 8);
    m.def("IOU_weighted_rotate_non_max_suppression_cpu", &IOU_weighted_rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou",
          "box"_a = 1, "box_corners"_a = 2,  "standup_iou"_a = 3, "thresh"_a = 4, "scores"_a = 5, "sigma"_a = 6, "labels"_a = 7,
          "dirs"_a = 8, "anchors"_a=9,"cnt_thresh"_a=10,"nms_sigma_dist_interval"_a=11,"nms_sigma_square"_a=12,
          "suppressed_thresh"_a=13,"centerness_c"_a=14);
    m.def("IOU_weighted_rotate_non_max_suppression_cpu", &IOU_weighted_rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou",
          "box"_a = 1, "box_corners"_a = 2,  "standup_iou"_a =3, "thresh"_a = 4, "scores"_a = 5, "sigma"_a = 6, "labels"_a = 7,
          "dirs"_a = 8, "anchors"_a=9, "cnt_thresh"_a=10,"nms_sigma_dist_interval"_a=11,"nms_sigma_square"_a=12,
          "suppressed_thresh"_a=13,"centerness_c"_a=14);
}