import os 

dataset_path = "/tmp/mask_data"

file_list = os.listdir(dataset_path)
total_num = len(file_list)
print(total_num)

train_num = int(total_num * 0.8)
print(train_num)

count = 0
for filename in file_list:
    file_path = os.path.join(dataset_path, filename)
    print(file_path)
    if count < train_num:
        if not os.path.exists(os.path.join(dataset_path, 'train')):
            os.makedirs(os.path.join(dataset_path, 'train'))
        new_file_path = os.path.join(dataset_path, 'train', filename)
        os.rename(file_path, new_file_path)
    else:
        if not os.path.exists(os.path.join(dataset_path, 'test')):
            os.makedirs(os.path.join(dataset_path, 'test'))
        new_file_path = os.path.join(dataset_path, 'test', filename)
        os.rename(file_path, new_file_path)

    
    count += 1

