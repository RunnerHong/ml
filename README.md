# ml

####1. 下载训练好的pth文件，放入weights文件夹下

####2. 在ml目录下执行，
 
    (1)、有cuda支持：
    ```
    python3 eval_Sepoch_for.py 
        --Anno_root "the_real_path_to_anno"
        --Image_root "the_real_path_to_image"
    ```
    (2)、无cuda支持（需装支持cpu的pytorch）：
    ```
    python3 eval_Sepoch_for.py 
        --Anno_root "the_real_path_to_anno"
        --Image_root "the_real_path_to_image"
        --cuda False
    ```