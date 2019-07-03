import cv2
import os

# pip install opencv-python
# pip install numpy


def get_photos():
    images_folder = 'images'
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), images_folder)
    if not os.path.exists(folder_path):
        os.mkdir(images_folder)
    print(folder_path)
    directions = ['up', 'down', 'left', 'right']

    # remove old photos
    for direction in directions:
        files_dir = os.path.join(folder_path, direction, '')
        if not os.path.exists(files_dir):
            os.mkdir(os.path.join(images_folder, direction))
        for file_path in os.listdir(os.path.join(images_folder, direction)):
            os.remove(os.path.join(images_folder, direction, file_path))

    vc = cv2.VideoCapture(0)
    count = 0
    dir_index = 0

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        raise Exception("Could not open video device")

    while rval:
        try:
            cv2.imshow(directions[dir_index], frame)
        except IndexError:
            cv2.destroyAllWindows()
            break
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        # print(key)
        if key == 13:  # take picture on ESC (27)
            file_path = os.path.join(folder_path, directions[dir_index], f'{directions[dir_index]}.{count}.png')
            cv2.imwrite(file_path, frame)  # save photos
            count += 1
        if key == 27:
            dir_index += 1
            count = 0
    cv2.destroyWindow("Camara")


class TestPhoto:
    def __init__(self, func):
        images_folder = 'temp'
        images_folder = os.path.join(images_folder, 'test_picture')
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), images_folder)
        if not os.path.exists(folder_path):
            os.mkdir(images_folder)
        print(folder_path)

        # remove data from directory
        for file_path in os.listdir(os.path.join(images_folder)):
            os.remove(os.path.join(images_folder, file_path))

        vc = cv2.VideoCapture(0)
        file_path = os.path.join(folder_path, 'temp_picture.png')

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
            while rval:
                cv2.imshow('Camera', frame)
                rval, frame = vc.read()
                key = cv2.waitKey(20)
                # print(key)
                cv2.imwrite(file_path, frame)  # save
                func()
                if key == 27:  # Exit on ESC
                    cv2.destroyAllWindows()
                    print('Proceso  detenido por el usuario')
                    break
        else:
            raise Exception("Could not open video device")


if __name__ == '__main__':
    get_photos()


