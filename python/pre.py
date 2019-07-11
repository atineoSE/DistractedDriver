import os, sys
from shutil import copyfile
from random import randrange

num_params = len(sys.argv)
sample = False
if num_params < 2:
    print("No arguments are passed. Using default.")
else:
    if num_params > 1:
        if sys.argv[1] == "sample":
            sample = True
            print("Use 10 imgs per class as sample.")

def get_distribution():
    r = randrange(101)
    return (r<75, r>=75 and r<80, r>=80)

base_src_path = "../data/imgs/train/"
if sample:
    base_dst_path = "../data/imgs/sample/"
    print("Copy sample data from {0} into {1}".format(base_src_path, base_dst_path))
else:
    base_dst_path = "../DistractedDriverCreateML/DistractedDriverCreateML/"
    print("Split data from {0} into Training, Validation and Test data under {1}".format(base_src_path, base_dst_path))

for root, dirs, files in os.walk(base_src_path):
    category_dir = os.path.basename(root)
    if category_dir == ".DS_Store":
        continue
    idx = 0
    for file in files:
        if sample:
            if idx > 9:
                break
            full_src_path = base_src_path + category_dir + "/" + file
            full_dst_path = base_dst_path + category_dir + "/" + file
            copyfile(full_src_path, full_dst_path)
            idx += 1
        else:
            (in_training_set, in_validation_set, in_test_set) = get_distribution()
            if in_training_set:
                split_dir = "TrainingData"
            elif in_validation_set:
                split_dir = "ValidationData"
            elif in_test_set:
                split_dir = "TestData"
            else:
                print("File does not belong to any set (training, validation, test)")
                quit()

            full_src_path = base_src_path + category_dir + "/" + file
            full_dst_path = base_dst_path + split_dir + "/" + category_dir + "/" + file
            #print("Copy from {0} to {1}".format(full_src_path, full_dst_path))
            copyfile(full_src_path, full_dst_path)

