import os
import shutil
import random

def get_file_list(input_dir, train_dir):
    file_list = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file)) and '.JPG' in file]
    t_list = os.listdir(train_dir)
    return list(set(file_list) - set(t_list))

def get_random_files(file_list, N):
    return random.sample(file_list, N)

def copy_files(random_files, input_dir, output_dir):
    for file in random_files:
        shutil.copy(os.path.join(input_dir, file), output_dir)

def main(input_dir, output_dir, train_dir):
    file_list = get_file_list(input_dir, train_dir)
    copy_files(file_list, input_dir, output_dir)

main('MorphProcessed/', 'test/0/', 'train/0/')
