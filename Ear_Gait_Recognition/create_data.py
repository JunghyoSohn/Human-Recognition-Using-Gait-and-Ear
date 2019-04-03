import os
import sys
from PIL import Image
import io
import random


def resize_pixel(image_file,outputfile, width_size, height_size):

    size = width_size, height_size
    for (path,dir,files) in os.walk(image_file):
        for filename in files:
            try:
                new_img = Image.new("RGB", (width_size, height_size), "black")

                fd = io.open(image_file+filename,'rb')
                im = Image.open(fd)
                im.thumbnail(size, Image.ANTIALIAS)

                load_img = im.load()
                load_newimg = new_img.load()

                i_offset = (width_size - im.size[0]) / 2
                j_offset = (height_size - im.size[1]) / 2

                for i in range(0, im.size[0]):
                    for j in range(0, im.size[1]):
                        load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                new_img.save(outputfile+filename,'JPEG')
                fd.close()

            except Exception as e:
                print("[Error]%s: Image writing error: %s" %(image_file+filename, str(e)))


def change_color(image_file, outputfile):

    for (path,dir,files) in os.walk(image_file):
        for filename in files:
            try:
                fd = io.open(image_file+filename,'rb')
                im = Image.open(fd).convert('LA')
                im.save(outputfile+filename+".png")

            except Exception as e:
                print("[Error]%s: Image writing error: %s" %(image_file, str(e)))

            fd.close()



def class_label(image_file, outputfile, cnt , kinds):
    if kinds=="Train":
        fd = io.open(outputfile+'Ear_train_data.txt', 'a')
    elif kinds=="Validation":
        fd = io.open(outputfile + 'Ear_validation_data.txt', 'a')

    for (path,dir,files) in os.walk(image_file):
        for filename in files:
            label_name = image_file.split('/')[-2]
            data = path+filename+","+ label_name + "," + str(cnt) + "\n"
            fd.write(data)
    fd.close()

def class_Test_label(image_file, outputfile):

    fd = io.open(outputfile + 'Ear_test_data.txt', 'w')

    for (path,dir,files) in os.walk(image_file):
        for filename in files:
            data = path+filename+ "\n"
            fd.write(data)
    fd.close()

def shuffle_label(label_file, outputfile):
    lines = open(label_file,'r').readlines()
    random.shuffle(lines)
    open(outputfile, 'w').writelines(lines)



if __name__ == "__main__":

    name=["Yoonsuk/","Hyejin/","Dongjun/","Donghun/","Junghyo/"]
    path_1="./Ear_image/Train/Before_preprocess/"
    path_2="./Ear_image/Train/After_preprocess/"
    path_t1="./Ear_image/Train/Before_preprocess/Validation_data/"
    path_t2="./Ear_image/Train/After_preprocess/Validation_data/"
    cnt=0

    for x in name:
        resize_pixel(path_1+x, path_1+"After_resize_pixel/"+x, 96,96)
        change_color(path_1+"After_resize_pixel/"+x, path_2+x)
        class_label(path_2+x, "./Train_Data/", cnt, "Train")

        # class_label(path_1 + "After_resize_pixel/" + x, "./Train_Data/", cnt)
        # class_label(path_1+x,"./Train_Data/", cnt, "Train")

        # validation
        resize_pixel(path_t1 + x , path_t1 + "After_resize_pixel/" + x, 96, 96)
        change_color(path_t1 + "After_resize_pixel/" + x, path_t2 + x)
        class_label(path_t2 + x, "./Train_Data/", cnt, "Validation")

        #class_label(path_t1 + x, "./Train_Data/", cnt, "Validation")
        # class_label(path_t1 + "After_resize_pixel/Test_data" + x, "./Train_Data/", cnt)
        cnt=cnt+1
