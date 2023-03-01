
- **The train and test for the DOTA dataset by SuperYOLO and GHOST.**
    ```python
    python train.py --data data/DOTA.yaml --cfg models/SRyolo_noFocus.yaml --ch 3 --input_mode RGB --batch-size 16 --epochs 100 --train_img_size 1024 --test_img_size 512 --device 0
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp2/weights/best.pt --full_weights runs/train/use/exp2/weights/best.pt --bit_width 32 --input_mode RGB 
    ```
    ```python
    python quantization_conv_automix_autodis.py --distillation 6 --inter_threshold 0.1 --device 0 --kd_weight 400 --epochs 100 --data data/DOTA.yaml --weights_teacher runs/train/exp2/weights/best.pt --weights runs/train/exp2/weights/best.pt --cfg models/SRyolo_noFocus.yaml --ch 3 --input_mode RGB --batch-size 16  --train_img_size 1024 --test_img_size 512 
    ```
    
    ```python
    python test.py --data data/DOTA.yaml --weights runs/train/exp3/weights/best.pt --batch-size 8 --save-conf --save-txt --device 0 --iou-thres 0.4
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp3/weights/best.pt --full_weights runs/train/use/exp2/weights/best.pt --input_mode RGB --inter-threshold 0.1
    ```
    ```python
    python data/DOTA_devkit_YOLO/ResultMerge.py --scratch runs/test/exp/labels/
    ```
    ```
    cd runs/test/exp/labels_merge/
    zip test.zip *.txt
    ```
    put the results (test.zip) to http://bed4rs.net:8001/evaluation2/
    
- **The train and test for the DIOR dataset by SuperYOLO and GHOST.**
    ```python
    python train.py --data data/Dior.yaml --cfg models/SRyolo_noFocus.yaml --ch 3 --input_mode RGB --batch-size 16 --epochs 150 --train_img_size 1024 --test_img_size 512 --device 0
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp4/weights/best.pt --full_weights runs/train/use/exp4/weights/best.pt --bit_width 32 --input_mode RGB 
    ```
    ```python
    python quantization_conv_automix_autodis.py --distillation 6 --inter_threshold 0.1 --device 0 --kd_weight 400 --epochs 150 --data data/Dior.yaml --weights_teacher runs/train/exp4/weights/best.pt --weights runs/train/exp4/weights/best.pt --cfg models/SRyolo_noFocus.yaml --ch 3 --input_mode RGB --batch-size 16 --train_img_size 1024 --test_img_size 512 
    ```
    ```python
    python test.py --data data/Dior.yaml --weights runs/train/exp5/weights/best.pt --batch-size 8 --device 0 --iou-thres 0.4
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp5/weights/best.pt --full_weights runs/train/use/exp4/weights/best.pt --input_mode RGB --inter-threshold 0.1
    ```
- **The train and test for the VEDAI dataset by SuperYOLO and GHOST.**
    ```python
    python train.py --data data/SRvedai.yaml --cfg models/SRyolo_noFocus_small.yaml --ch 4 --input_mode RGB+IR --batch-size 2 --epochs 300 --train_img_size 1024 --test_img_size 512 --device 0 --input_mode RGB+IR
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp6/weights/best.pt --full_weights runs/train/use/exp6/weights/best.pt --bit_width 32 --input_mode RGB+IR
    ```
    ```python
    python quantization_conv_automix_autodis.py --distillation 6 --inter_threshold 0.1 --device 0 --kd_weight 400 --epochs 300 --data data/SRvedai.yaml --weights_teacher runs/train/exp6/weights/best.pt --weights runs/train/exp6/weights/best.pt --cfg models/SRyolo_noFocus_small.yaml --ch 4 --input_mode RGB+IR --batch-size 2 --train_img_size 1024 --test_img_size 512 
    ```
    ```python
    python test.py --data data/SRvedai.yaml --weights runs/train/exp7/weights/best.pt --batch-size 1 --device 0 --iou-thres 0.6
    ```
    ```python
    python test_flops.py --weights runs/train/use/exp7/weights/best.pt --full_weights runs/train/use/exp6/weights/best.pt --input_mode RGB+IR --inter-threshold 0.1
    ```
