import numpy as np
import cv2 as cv
import shutil
import glob
import os
#załaduj obraz z folderu w pętli
# sprawdź czy obraz zawiera jakiś procent danych kolorów
    # - zielony, biały, czerwony, niebieski, zielony
#funkcja zwracająca maskę danego koloru
#sprawdzenie ile jest procentowo obszaru maski
#Zdecydowanie czy obraz należy usuunąć 

def click_event(event, x, y, flags, params): 
    if event==cv.EVENT_LBUTTONDOWN: 
        font = cv.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h = img_hsv[y, x, 0] 
        s = img_hsv[y, x, 1] 
        v = img_hsv[y, x, 2] 
        # print(f'[blue : {b}, green : {g}, reed : {r}]')
        print(f'[hue : {h}, saturation : {s}, value : {v}]')

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        cv.imshow('image', img) 

def get_hsv_mask(image, lower_hsv, upper_hsv):    
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask

def reject_image(image: cv.UMat, verbosity: int = 0) -> bool:
    #Colors to search for
    yellow_lower = (20, 90, 145)
    yellow_upper = (25, 120, 178)

    white_lower = (100, 0, 180)
    white_upper = (120, 30, 220)

    green_lower = (42, 80, 87)
    green_upper = (61, 240, 230)

    red_lower = (20, 90, 145)
    red_upper = (25, 107, 170)

    yellow_mask = get_hsv_mask(image, yellow_lower, yellow_upper)
    white_mask = get_hsv_mask(image, white_lower, white_upper)
    red_mask = get_hsv_mask(image, red_lower, red_upper)
    green_mask = get_hsv_mask(image, green_lower, green_upper)

    mask = yellow_mask + white_mask + green_mask
    mask_pixels = np.count_nonzero(mask)
    image_pixels = image.shape[0]*image.shape[1]
    procentage = (mask_pixels/image_pixels)*100
    
    if verbosity != 0:
        print(f'mask pixels prctntage: {procentage:.2f}%')
        cv.imshow('Image with mask', cv.bitwise_and(image, image, mask=mask))
        cv.resizeWindow('Image with mask', 200, 200)
    
    if procentage >= 20:
        return True
    return False

def move_file(file_name: str, file_src: str, target_dir: str) -> None:
    src_dir = os.path.join(file_src, file_name)
    target_dir = os.path.join(target_dir, file_name)
    try:
        shutil.move(src_dir, target_dir)
        print(f"Plik {file_name} został przeniesiony z {file_src} do {target_dir}.")
    except FileNotFoundError:
        print(f"Plik {file_name} nie istnieje w folderze źródłowym.")
    except Exception as e:
        print(f"Wystąpił błąd podczas przenoszenia pliku: {str(e)}")

def main():
    # wczytanie obrazu
    folders_dir = [r'samples\dab', r'samples\sosna']
    for folder_dir in folders_dir:
        for image_path in glob.iglob(f'{folder_dir}/*'):
            if (image_path.endswith('.png') or image_path.endswith('.jpg')):
                print(image_path)
                # image_path = r'samples\dab\dab_44_155.jpg'
                global img 
                global img_hsv
                img = cv.imread(image_path)

                if reject_image(img) == True:
                    src_path = image_path.split('\\')
                    file_name = str(src_path[-1])
                    src_dir = '\\'.join(src_path[:-1])
                    target_dir = f'samples\\rejected samples\\{src_path[1]}'
                    move_file(file_name, src_dir, target_dir)

            # cv.imshow('image', img) 
            # cv.resizeWindow('image', 200, 200) 
            # cv.setMouseCallback('image', click_event) 
            # key = cv.waitKey(0)
            # if key == ord('Q') or key == ord('q') or key == 27:
            #     cv.destroyAllWindows() 
            #     exit()

if __name__ == '__main__':
    main()