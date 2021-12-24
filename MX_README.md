##SE-SSD_CLOCs_fusion  

### Train
SECOND_based
```bash
$ cd SE-SSD  
$ python tools/MX_fusion_train.py --config=examples/second/configs/MX_fusion_train_config.py --checkpoint=epoch_60.pth
```
pointpillars_based
```bash
$ cd SE-SSD  
$ python tools/MX_fusion_train.py --config=examples/point_pillars/configs/MX_fusion_train_config.py --checkpoint=epoch_60.pth
```

###Eval  
SECOND_based
```bash
$ cd SE-SSD  
$ python tools/MX_fusion_test.py --config=examples/second/configs/MX_fusion_test_config.py --checkpoint=epoch_60.pth
```
pointpillars_based
```bash
$ cd SE-SSD  
$ python tools/MX_fusion_test.py --config=examples/point_pillars/configs/MX_fusion_test_config.py --checkpoint=epoch_60.pth
```


### 2021-12-24
SECOND_based (fusion_layer-40832.tckpt)
```text
Evaluation official_AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.61, 95.94, 93.52
bev  AP:96.82, 92.39, 90.04
3d   AP:94.02, 86.49, 83.93
aos  AP:99.59, 95.74, 93.14
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:99.61, 95.94, 93.52
bev  AP:99.61, 96.19, 95.90
3d   AP:99.80, 96.11, 93.65
aos  AP:99.59, 95.74, 93.14
```
