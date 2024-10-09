# A Novel Edge Detection Method of Blade with Multi-Supervision for Fore-Background Confusion Caused by Extreme Illumination
![graph_abstract](BED_illu/Graphical_Abstract.jpg)
![pipeline](BED_illu/pipeline.jpg)

## train


1. Unzip the dataset to the `dataset/` directory.
2. Modify lines 47 and 48 in the `TrainData.py` file to select the desired data partitioning.
    ```
        partitioning I
        self.dataEdge = osp.join('datalst', 'HSVEnhance', 'train_pair_root_HSV_ydd.txt')
        self.dataSOD = osp.join('datalst', 'HSVEnhance', 'train_pair_root_sod_HSV_ydd.txt')
   
        partitioning II
        self.dataEdge = osp.join('datalst', 'HSVEnhance', 'train_pair_root_HSV_ycv2.txt')
        self.dataSOD = osp.join('datalst', 'HSVEnhance', 'train_pair_root_sod_HSV_ycv2.txt')
    
    ```
   
3. run `main.py` to train the model.
    ```
        # test(config)
        train(config)
    ```

## test

1. Fill in the parameters in the `args.py` file.
    ```
    // checkpoint path
    parser.add_argument('--model', type=str, default='checkpoint/ycv2/ckpt_60.pt')
    // path to save .png results
    parser.add_argument('--test_fold', type=str, default='Predictions/ycv2_sal')
    // path to save .mat results
    parser.add_argument('--mat_fold', type=str, default='Predictions/ycv2_sal_mat')
    // 1 for SOD 0 for ED
    parser.add_argument('--test_mode', type=int, default=0)  # choose task
    // choose test dataset
    parser.add_argument('--sal_mode', type=str, default='p1')
    
    ```
2. run `main.py` to test the model.
    ```
	test(config)
	# train(config)
    ```


## Result
Detecting edges under extreme lighting conditions. (**A BETTER RESULTS ONLY IF  TRAINING ON SELF ANNOTATED DATA**)
![biped](BED_illu/biped_gen.jpg)

![ours](BED_illu/Ourvisible.jpg)
## Citation
```
@ARTICLE{10623389,
  author={An, Chenxi and Song, Kechen and Bao, Lulu and Zhao, Dong and Zhou, Zhenbo and Yan, Yunhui},
  journal={IEEE Sensors Journal}, 
  title={A Novel Edge Detection Method of Blade with Multi-Supervision for Fore-Background Confusion Caused by Extreme Illumination}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Image edge detection;Task analysis;Lighting;Feature extraction;Blades;Object detection;Optical variables measurement;Aeroengine blade (AEB);edge detection;vision-based measurement (VBM);high-level feature learning;extreme illumination},
  doi={10.1109/JSEN.2024.3435358}}
  
```
