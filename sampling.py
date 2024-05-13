import cv2 as cv
import os

def main():
    save = True
    # crop_Tree()
    drzewo = 'sosna'
    # drzewo = 'dab'
    sampling(drzewo, save)

def crop_Tree(save_output=False):
    # for i in range(4,114):
    for i in range(1,90):
        # path_to_mask = fr'maski\sosna\newmask-sosna_({i}).jpg'
        # path_to_image = fr'resources\drzewa\sosna\sosna_({i}).jpg'
        path_to_mask = fr'maski\dab\newmask-dab_({i}).jpg'
        path_to_image = fr'resources\drzewa\dab\dab_({i}).jpg'

        mask = cv.imread(path_to_mask, cv.IMREAD_GRAYSCALE)
        img = cv.imread(path_to_image)

        x, y, w, h = cv.boundingRect(mask)
        final = img[y:y+h, x:x+w]
        cv.imshow("final", final)
        key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            exit()
        cv.destroyAllWindows()
        if save_output == True:
            output_file_name = 'new_' + path_to_image.split('\\')[-1]
            output_diectory = ('kora\\' + path_to_image.split('\\')[-2]) + '\\'
            cv.imwrite(output_diectory+output_file_name, final)


def sampling(drzewo, save_output=False):
    directory = 'kora\sosna' if drzewo == 'sosna' else 'kora\dab'
    for filename in os.listdir(directory):
        path_to_image = os.path.join(directory, filename)
        if os.path.isfile(path_to_image):
            # print(path_to_image)
            start_index = filename.find('(') + 1
            end_index = filename.find(')', start_index)
            if start_index != -1 and end_index != -1:
                number = int(filename[start_index:end_index])
            img = cv.imread(path_to_image)
            sample_size = 50
            offset = 30
            x, y = 0, 0
            n= 0 
            while x+sample_size < img.shape[1]:
                while y+sample_size < img.shape[0]:
                    crop_img = img[y:y+sample_size, x:x+sample_size]
                    # img = cv.rectangle(img,(x,y),(x+sample_size,y+sample_size),(0,0,255),1)
                    # cv.imshow('cropped', img)
                    # key = cv.waitKey(0)
                    # if key == ord('q') or key == ord('Q') or key == 27:
                    #     exit()
                    # cv.destroyAllWindows() 

                    if save_output == True:
                        output_file_name =  path_to_image.split('\\')[-2] + "_"+ str(number) + "_" + str(n) + ".jpg"
                        output_diectory = ('samples\\' + path_to_image.split('\\')[-2]) + '\\'
                        cv.imwrite(output_diectory+output_file_name, crop_img)
                        # print(output_diectory)
                        # print(output_file_name)
                    n+=1
                    y+=offset
                y = 0
                x+=offset

        # cv.imshow('cropped', crop_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


    



if __name__ == "__main__":
    main()