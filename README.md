#The documentation of active_few_shot_learning

Libraries:
use miniconda3 environment
1. python 3.5.4
2. tensorflow1.4 gpu
3. cuda8.0
4. cudnn 7.0.5
5. numpy 1.14.0
6. Pillow 5.0.0
7. tqdm
8. pandas 0.22.0

step 1: run mini_data_construct.py to sample the data
step 2: run data_augmentation.py to augment the tra_query_data
step 3: run train_actively.py to train the network

info: you can easily understand the variable via name
and feel free to change the batch_size, query_size, support_image_number(shots)
active_round etc.



