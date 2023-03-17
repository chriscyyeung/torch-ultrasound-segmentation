call c:/Users/Chris/anaconda3/Scripts/activate.bat c:/Users/Chris/anaconda3/
call activate torch-gpu
@echo off
python train.py --train_img_dirs d:\Chris\PICAI\data\picai_public_images_fold1 --val_img_dir d:\Chris\PICAI\data\picai_public_images_fold3 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type t2w --epochs 20 --batch_size 8
python train.py --train_img_dirs d:\Chris\PICAI\data\picai_public_images_fold1 --val_img_dir d:\Chris\PICAI\data\picai_public_images_fold3 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type hbv --epochs 20 --batch_size 8
python train.py --train_img_dirs d:\Chris\PICAI\data\picai_public_images_fold1 --val_img_dir d:\Chris\PICAI\data\picai_public_images_fold3 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type adc --epochs 20 --batch_size 8
call conda deactivate