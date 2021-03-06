import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageEnhance
import os
import glob

path = os.getcwd()
csv_dir = path + '/miniImagenet/csv/'


def _read_image_as_array(image, dtype='float32'):
    # f = Image.open(image)
    f = image
    k = np.random.randint(0, 4)
    f.rotate(k*90)
    f = f.filter(ImageFilter.GaussianBlur(1.5))
    f = random_brightness(f, [0.7,1.3])
    try:
        #image = np.asarray(f, dtype=dtype)
        image = tf.keras.preprocessing.image.img_to_array(f)
        #print(image.shape)
        image = augment_image(image)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image
    #p = Augmentor.Pipeline(image)
    #p.rotate(probability=0.7, max_left=20,max_right=20)
    #p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    #p.flip_left_right(probability=0.5)
    #p.crop_center(probability=0.8, percentage_area=0.3)
    #p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=3)
    #p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)


def augment_image(img):

    img = tf.keras.preprocessing.image.random_rotation(img, 30, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shear(img, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_zoom(img, [0.9, 0.9], row_axis=0, col_axis=1, channel_axis=2)

    return img


def random_brightness(img, brightness_range):

    imgenhancer_Brightness = ImageEnhance.Brightness(img)

    u = np.random.uniform(brightness_range[0], brightness_range[1])

    img = imgenhancer_Brightness.enhance(u)

    return img


def move_with_augmentation(r, c, imgs):

    destination_dir = path + '/miniImagenet/tra_support_data_mini/'
    for i, img in enumerate(imgs):
        original_img = Image.open(img)
        f = original_img
        for j in range(1):
            if j == 0:
                new_name = destination_dir+str(c)+'/'+str(i)+'_round_'+str(r)+'.png'
                os.rename(img, new_name)
            else:
                new_aug_name = destination_dir+str(c)+'/'+str(i)+'_round_'+str(r)+'_aug_'+str(j-1)+'.png'
                new_aug_img = _read_image_as_array(f)
                aug_img = Image.fromarray(new_aug_img.astype('uint8'))
                aug_img.save(new_aug_name)

        if hasattr(original_img, 'close'):
            original_img.close()

def tra_data_augmentation(dir):

    folders = os.listdir(dir)
    for folder in folders:
        print(folder)
        img_dir = dir + str(folder)
        imgs = glob.glob(img_dir + '/*.png')
        for i in range(4):
            for img in imgs:
                f = Image.open(img)
                aug_img_name = img[:-4] + '_aug_' + str(i) + '.png'
                aug_img = _read_image_as_array(f)
                aug_img = Image.fromarray(aug_img.astype('uint8'))
                aug_img.save(aug_img_name)


if __name__ == '__main__':

    #dir = path + '/miniImagenet/tra_support_data_mini/'
    #tra_data_augmentation(dir)
    dir_1 = path + '/miniImagenet/tra_query_data_mini/'
    tra_data_augmentation(dir_1)
