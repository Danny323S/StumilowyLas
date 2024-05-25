import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import cv2 as cv
import tensorflow as tf

def get_model(disp_summary=0):
    #Tworzenie modelu
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
        ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   
    if disp_summary != 0:
        model.summary()

    return model


def load_data(folder_dir: str) -> list[np.array] | list[int]:
    try:
        image_files = [
            os.path.join(folder_dir, file)
            for file in os.listdir(folder_dir) if file.lower().endswith('.jpg')]
        labels = [
            0 if file.split('_')[0] == 'dab' else 1
            for file in os.listdir(folder_dir) if file.lower().endswith('.jpg')]     
    except Exception as e:
        print(f"Błąd podczas wczytywania plików .jpg z folderu {folder_dir}: {e}")

    images_list = []
    for img_file in image_files:
        try:
            img = cv.imread(img_file)
            images_list.append(img)
        except Exception as e:
             print(f"Błąd podczas wczytywania obrazu {img_file}: {e}")

    return images_list, labels


def left_right_augmentation(input_images: list[np.array], num_to_augment) -> list[np.array]:
    flipped =[]
    random.shuffle(input_images)
    for i in range(num_to_augment):
        cv.imshow("przed",input_images[i])
        flipped.append(np.fliplr(input_images[i]))
        cv.imshow("po",flipped[i])
        cv.waitKey(0)
    cv.destroyAllWindows()
    return flipped

def save_flipped(images: list[np.array], path:str):
    for i in range(len(images)):
        filepath = os.path.join(path , 'dab_augmented_'+ str(i)+ '.jpg')
        # print(filepath)
        cv.imwrite(filepath, images[i])

        
def main():
    # pobranie modelu sieci cnn
    # model = get_model(disp_summary=1)
    # print(f'model type: {type(model)}')

    # wczytanie danych do uczenia sieci
    print('Loading data')
    paths = [
        'samples\\data\\train\\dab',
        'samples\\data\\val_test\\dab',
        'samples\\data\\train\\sosna',
        'samples\\data\\val_test\\sosna'
    ]   

    dab_train_images, dab_train_labels = load_data(paths[0])
    dab_test_images, dab_test_labels = load_data(paths[1])
    sosna_train_images, sosna_train_labels = load_data(paths[2])
    sosna_test_images, sosna_test_labels = load_data(paths[3])


    # # Zliczenie ile danych trzeba dorobić żeby zbiory były równe

    num_augm_dab_train = len(sosna_train_images)-len(dab_train_images)
    num_augm_dab_test = len(sosna_test_images)-len(dab_test_images)
    print("Do treningowego " + str(num_augm_dab_train))
    print("Do testowego " + str(num_augm_dab_test))

    # Lustrzane odbicia w pionie
    flipped_dab_train = left_right_augmentation(dab_train_images, num_augm_dab_train)
    flipped_dab_test = left_right_augmentation(dab_test_images, num_augm_dab_test)
    print(len(flipped_dab_test))

    path_augmented_train = rf'C:\Users\niepo\OneDrive\Dokumenty\GitHub\StumilowyLas\samples\data\augmented\train'
    path_augmented_test = rf'C:\Users\niepo\OneDrive\Dokumenty\GitHub\StumilowyLas\samples\data\augmented\val_test'
    # save_flipped(flipped_dab_train, path_augmented_train)
    # save_flipped(flipped_dab_test, path_augmented_test)

    #załadowanie augmentowanych danych
    # dab_augm_train_images, dab_augm_train_labels = load_data(path_augmented_train)
    # dab_augm_test_images, dab_augm_test_labels = load_data(path_augmented_test)

    # Tutaj łączę dęby i sosny w dwa zbiory, terningowe i testowe.
    # następnie mieszam te  zbiory tak tak aby w zbiorze dane nie były uporządkowane,
    # czyli aby w zbiorach nie były najpierw same dęby a później same sosny (nie wiem czy jest to konieczne)
    # training_data = list(zip(dab_train_images + dab_augm_train_images+ sosna_train_images, dab_train_labels +dab_augm_train_labels+ sosna_train_labels))
    # random.shuffle(training_data)
    # training_images, training_labels = zip(*training_data)
    # training_images, training_labels = list(training_images), list(training_labels)

    # test_data = list(zip(dab_test_images + dab_augm_test_images + sosna_test_images, dab_test_labels + dab_augm_test_labels + sosna_test_labels))
    # random.shuffle(test_data)
    # test_images, test_labels = zip(*test_data)
    # test_images, test_labels = list(test_images), list(test_labels)
    # print('Loading data finished')

    # model.fit(np.array(training_images), np.array(training_labels), epochs=10)
    # test_loss, test_accuracy = model.evaluate(np.array(test_images), np.array(test_labels))
    # print ('\n Validation results:\naccuracy: {}%,  loss: {}'.format(test_accuracy*100, test_loss))

    # zapis modelu
    save = False
    if save == True:
        target_folder = 'saved models'
        file_name = 'model_10epochs'
        extension = 'keras'
        path = f'{target_folder}\\{file_name}.{extension}'
        print(f'saving model to: {path}')
        model.save(path)

if __name__ == '__main__':
    main()