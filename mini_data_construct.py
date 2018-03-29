import os
import numpy as np
import pandas as pd
from collections import defaultdict
import tarfile
from PIL import Image
import glob

data_path = '/home/xiangwei/active_few_shot_learning/'

path = os.getcwd()

csv_path = data_path + '/miniImagenet/csv/'
tar_path = data_path + '/miniImagenet/'

if not os.path.exists(path + '/miniImagenet'):
    os.makedirs(path + '/miniImagenet')

data_dir = path + '/miniImagenet/'

img_number = defaultdict(list)

_NUM_TRA_SUPPORT = 20
_NUM_TRA_QUERY = 500
_NUM_VAL = 500
_NUM_TEST = 500
_NUM_CAND_TRA = 80


def find_five_candidates():
    res = []
    csv_dir = csv_path + 'tra.csv'
    csv = pd.read_csv(csv_dir, sep=',')
    labels = csv.label.unique().tolist()

    for k, label in enumerate(labels):
        tar = tarfile.open(tar_path + label + '.tar')
        imgs = tar.getmembers()

        img_number[label].append(len(imgs))

    selected_labels = sorted(img_number.items(), key=lambda x: x[1],  reverse=True)

    for c in selected_labels[3:8]:
        label = c[0]
        res.append(label)

    return res


def extract_file(res):

    for label in res:
        tar = tarfile.open(tar_path + label + '.tar')
        if not os.path.exists(data_dir + label):
            os.makedirs(data_dir + label)

        tar.extractall(data_dir + label)
        tar.close()


def construct_data(res):
    if not os.path.exists(data_dir + 'tra_support_data_mini'):
        os.makedirs(data_dir + 'tra_support_data_mini')

    if not os.path.exists(data_dir + 'tra_query_data_mini'):
        os.makedirs(data_dir + 'tra_query_data_mini')

    if not os.path.exists(data_dir + 'val_data_mini'):
        os.makedirs(data_dir + 'val_data_mini')

    if not os.path.exists(data_dir + 'test_data_mini'):
        os.makedirs(data_dir + 'test_data_mini')

    if not os.path.exists(data_dir + 'tra_cand_data_mini'):
        os.makedirs(data_dir + 'tra_cand_data_mini')

    for i, label in enumerate(res):
        if not os.path.exists(data_dir + 'tra_support_data_mini/' + str(i)):
            os.makedirs(data_dir + 'tra_support_data_mini/'+ str(i))

        if not os.path.exists(data_dir + 'tra_query_data_mini/'+ str(i)):
            os.makedirs(data_dir + 'tra_query_data_mini/'+ str(i))

        if not os.path.exists(data_dir + 'val_data_mini/' + str(i)):
            os.makedirs(data_dir + 'val_data_mini/' + str(i))

        if not os.path.exists(data_dir + 'test_data_mini/' + str(i)):
            os.makedirs(data_dir + 'test_data_mini/' + str(i))

        if not os.path.exists(data_dir + 'tra_cand_data_mini/' + str(i)):
            os.makedirs(data_dir + 'tra_cand_data_mini/' + str(i))

    for j, label in enumerate(res):

        img_dir = data_dir + label
        imgs = glob.glob(img_dir + '/*.JPEG')
        print(len(imgs))
        for m, img in enumerate(imgs):
            try:
                f = Image.open(img)
                f = f.resize((224,224))
                f = np.asarray(f, dtype=np.float32)
                f = np.reshape(f, (224, 224, 3))
                f = Image.fromarray(f.astype('uint8'))
                if m < _NUM_TRA_SUPPORT:
                    img_name = data_dir + 'tra_support_data_mini/' + str(j) + '/' + str(m) + '.png'
                    f.save(img_name)
                elif _NUM_TRA_SUPPORT <= m < _NUM_CAND_TRA + _NUM_TRA_SUPPORT:
                    img_name = data_dir + 'tra_cand_data_mini/' + str(j) + '/' + str(m - _NUM_TRA_SUPPORT) + '.png'
                    f.save(img_name)
                    img_name = data_dir + 'tra_query_data_mini/' + str(j) + '/' + str(m - _NUM_TRA_SUPPORT) + '.png'
                    f.save(img_name)
                elif _NUM_TRA_SUPPORT + _NUM_CAND_TRA <= m < _NUM_TRA_SUPPORT + _NUM_TRA_QUERY + _NUM_CAND_TRA:
                    img_name = data_dir + 'tra_query_data_mini/' + str(j) + '/' + str(m - _NUM_TRA_SUPPORT) + '.png'
                    f.save(img_name)
                    # m += 1
                elif _NUM_TRA_SUPPORT + _NUM_TRA_QUERY + _NUM_CAND_TRA <= m < _NUM_TRA_SUPPORT  + _NUM_TRA_QUERY + _NUM_CAND_TRA + _NUM_VAL:
                    img_name = data_dir + 'val_data_mini/' + str(j) + '/' + str(m - _NUM_TRA_SUPPORT -_NUM_TRA_QUERY - _NUM_CAND_TRA) + '.png'
                    f.save(img_name)
                    # m += 1
                else:
                    img_name = data_dir + 'test_data_mini/' + str(j) + '/' + str(m - _NUM_TRA_SUPPORT - _NUM_VAL - _NUM_TRA_QUERY - _NUM_CAND_TRA) + '.png'
                    f.save(img_name)
                    if m == 1599:
                        break

            except Exception as e:
                print('skipping image, beacuse ' + str(e))


def delect_files():
    files = glob.glob(path + '/*.JPEG')
    for file in files:
        os.remove(file)


if __name__ == '__main__':

    res = find_five_candidates()
    extract_file(res)
    construct_data(res)
    # delect_files()
