## SECOND_based_SE-SSD_CLOCs_fusion  

Train
```bash
$ cd SE-SSD  
$ python tools/MX_sec_fusion_train.py --config=examples/second/configs/MX_fusion_train_config.py --checkpoint=epoch_60.pth

```

Eval
```bash
$ cd SE-SSD  
$ python tools/MX_sec_fusion_test.py --config=examples/second/configs/MX_fusion_test_config.py --checkpoint=epoch_60.pth
```