# ml

1. 下载训练好的pth文件，放入weights文件夹下

2. 准备好数据，数据目录下有两个文件夹，Image和Annotation。
    ```
    /foo
        /Image
        /Annotation
    ```    

3. 在ml目录下执行， python3 eval_Sepoch_for.py --SIXray_root "the_real_path_to/foo"