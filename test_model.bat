call c:/Users/Chris/anaconda3/Scripts/activate.bat c:/Users/Chris/anaconda3/
call activate torch-gpu
@echo off
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type t2w --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_t2w_model_20230320_212039.pt
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type hbv --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_hbv_model_20230320_232634.pt
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type adc --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_adc_model_20230321_012953.pt
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type t2w --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_t2w_model_20230320_143309.pt
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type hbv --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_hbv_model_20230320_164025.pt
python evaluate.py --test_img_dir d:\Chris\PICAI\data\picai_public_images_fold0 --mask_dir d:\Chris\PICAI\data\picai_labels\ --img_type adc --model_path c:\Users\Chris\Documents\CISC881\prostate-cancer-segmentation\TrainedModels\best_adc_model_20230320_185116.pt
call conda deactivate