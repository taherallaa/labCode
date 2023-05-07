
import cv2
import numpy as np


counter = 0
P = []
T = []
weights = np.array([])
bias = np.array([])
image = None
text = "Type is : ???"


def start():
    global weights, T, P

    for i in range(11):
        P.append( get_feature(cv2.imread(f"data2/car.{i}.jpeg",cv2.IMREAD_GRAYSCALE)) )
        T.append(1)
        
        P.append( get_feature(cv2.imread(f"data2/plane.{i}.jpeg",cv2.IMREAD_GRAYSCALE)) )
        T.append(-1)

    P = np.array(P)
    T = np.array(T)

    weights = np.dot(T, np.dot(np.linalg.inv(np.dot(P, P.T)), P))
    print(weights)

# ...and define functions that return data to be displayed...

def get_feature(image):
    new = conv_relu(image)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = flatten(new)
    return new

def get_image():
    if(image == None):
        return f"data2/plane.0.jpeg"
    
    return image.__dict__['url']

# ...or that handle user input

def conv_relu(image):
    mask = [[-1,-1,1],[0,1,-1],[0,1,1]]
    size1 = len(image) - 2
    size2 = len(image[0]) - 2
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(size1):
        for j in range(size2):
            x = 0
            for k in range(3):
                x += (image[i+k][j+0]*mask[k][0] + image[i+k][j+1]*mask[k][1] + image[i+k][j+2]*mask[k][2])
            new_image[i][j] = x if x > 0 else 0

    return new_image

def pooling(image):
    size1 = int(len(image)/2)
    size2 = int(len(image[0])/2)
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(0,size1):
        for j in range(0,size2):
            x = 0
            for k in range(2):
                x += (image[(i*2)+k][(j*2)+0] + image[(i*2)+k][(j*2)+1])/4
            new_image[i][j] = int(x)
    
    return new_image

def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(el)
    return new_image

def neural():
    global image, weights, text
    p = np.array(get_feature(cv2.imread(f'data2/' + image.__dict__['name'],cv2.IMREAD_GRAYSCALE)))
    p = p.T
    print(p)
    n = np.dot(weights, p)
    print(n)
    text = "Type is : Car" if n >=0 else "Type is : Plane"

