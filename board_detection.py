import cv2
import matplotlib.pyplot as plt

nline = 11
ncol = 9

if __name__ == "__main__":
    image = cv2.imread("./data/images/whole_game/OwnGame20.png")

    plt.imshow(image)
    plt.show()