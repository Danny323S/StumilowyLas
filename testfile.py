import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2 as cv
import numpy as np

def main():
    paths = ['samples\\dab-test\\dab_3_2.jpg',
            'samples\\dab-test\\dab_3_3.jpg']

    size = (50, 50)
    label = 0

    images_list = []
    for path in paths:
        try:
            img = cv.imread(path)
            # img = np.array(img)
            images_list.append(img)
        except Exception as e:
            print(f'Błąd wczytywania obrazu {path}: {e}')
    
    images_list = np.array(images_list)

    print(f'image type: {type(img)}')
    print(f'image shape: {(img.shape)}')
    print(f'image number of dimensions: {np.ndim(img)}')
    print(f'image data type: {img.dtype}')

    for img in images_list:
        cv.imshow(str(label), img)

        key = cv.waitKey(0)
        if key == ord('q'):
            exit()


if __name__ == '__main__':
    main()