import cv2
import os

def main():
    file_names = [f for f in os.listdir(folder)]
    for name
    gt = cv2.imread('./dataset/gt/' + name)[:, :, ::-1]
    diffuser = cv2.imread('./dataset/diffuser/' + name)[:, :, 0]


if __name__ == '__main__':
    main()