import cv2
import numpy as np
from scipy.ndimage.filters import convolve
import sys

FORWARD_ENERGY = 1

"""
input: an image
output: the same image rotated by 90 degrees
"""
def rotate(image):
    new_image = np.zeros((image.shape[1],image.shape[0],3))
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i][j] = image[j][i]
            
    return new_image

"""
input: an image
output: a 2D array representing the backward energy map of the image
"""
def backward_energy(image):
    image = image.sum(axis=2)

    kernel_x = [[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    kernel_y = [[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]
    
    energy = np.absolute(convolve(image, kernel_x)) + np.absolute(convolve(image, kernel_y))

    return energy

"""
input: an image 
output: a 2D array representing the forward energy map of the image
"""
def forward_energy(image):
    x = image.shape[0]
    y = image.shape[1]
    
    image = image.sum(axis=2)
    
    up = image
    up = np.insert(up, 0, up[x-1], 0)
    up = np.delete(up, x, 0)
    
    left = up
    left = np.insert(left, 0, up[:,y-1], 1)
    left = np.delete(left, y, 1)
    
    right = up
    right = np.insert(right, y, up[:,0], 1)
    right = np.delete(right, 0, 1)
    
    Cup = np.abs(right - left)
    Cleft = np.abs(up - left) + Cup
    Cright = np.abs(up - right) + Cup
    
    energy = np.zeros((x, y))
    m = np.zeros((x, y))

    for i in range(1, x):
        Mup = m[i-1]
        
        Mleft = Mup
        Mleft = np.insert(Mleft, 0, Mup[y-1])
        Mleft = np.delete(Mleft, y)
        
        Mright = Mup
        Mright = np.insert(Mright, y, Mup[0])
        Mright = np.delete(Mright, 0)
        
        C = np.array((Cup[i], Cleft[i], Cright[i]))
        M = np.array((Mup + Cup[i], Mleft + Cleft[i], Mright + Cright[i]))

        energy[i] = np.choose(np.argmin(M, axis=0), C)
        m[i] = np.choose(np.argmin(M, axis=0), M)
        
    return energy

"""
input: an image
output: the minimum seam of the image
"""
def minimum_seam(image):
    x = image.shape[1]
    y = image.shape[0]
    
    seam = np.zeros((x,y))
    energy_sum = np.zeros((x))
    
    if FORWARD_ENERGY == 1:
        map = forward_energy(image)
    else:
        map = backward_energy(image)

    for i in range(x):
        for j in range(y):
            if j == 0:
                seam[i][j] = i
            else:
                index = int(seam[i][j-1])
                if index == 0:
                    if map[j][index] < map[j][index+1]:
                        seam[i][j] = index
                        energy_sum[i] += map[j][index]
                    else:
                        seam[i][j] = index+1
                        energy_sum[i] += map[j][index+1]
                elif index == x-1:
                    if map[j][index] < map[j][index-1]:
                        seam[i][j] = index
                        energy_sum[i] += map[j][index]
                    else:
                        seam[i][j] = index-1
                        energy_sum[i] += map[j][index-1]
                else:
                    if map[j][index] <= min(map[j][index-1],map[j][index+1]):
                        seam[i][j] = index
                        energy_sum[i] += map[j][index]
                    elif map[j][index-1] < map[j][index+1]:
                        seam[i][j] = index-1
                        energy_sum[i] += map[j][index-1]
                    else:
                        seam[i][j] = index+1
                        energy_sum[i] += map[j][index+1]

    return seam[np.argmin(energy_sum)]

"""
input: an image
output: same image but with a seam removed, and the seam removed
"""
def remove_column(image):
    x = image.shape[0]
    y = image.shape[1]
        
    seam = minimum_seam(image)
    new_image = np.zeros((x,y-1,3))
        
    for i in range(x):
        temp = 0
        for j in range(y-1):
            new_image[i][j] = image[i][j+temp]
            if j == int(seam[i]):
                temp = 1
            
    return new_image,seam

"""
input: an image, and the number of seams to be removed
output: same image but with seams removed
"""
def remove_columns(image,number):
    for h in range(number):
        sys.stdout.write("\r{0}%".format(int(h/number*100)+1))
        sys.stdout.flush()
        image,_ = remove_column(image)
        
    return image

"""
input: an image, and a seam to be added
output: same image but with a seam added
"""
def add_column(image,seam):
    x = image.shape[0]
    y = image.shape[1]
    
    new_image = np.zeros((x,y+1,3))
    
    for i in range(x):
        temp = 0
        for j in range(y):
            new_image[i][j+temp] = image[i][j]
            if j == seam[i]:
                temp = 1
                if i == 0:
                    new_image[i][j+1] = image[i][j]*0.35 + image[i][j+1]*0.35 + image[i+1][j]*0.15 + image[i+1][j+1]*0.15
                elif i == x-1:
                    new_image[i][j+1] = image[i][j]*0.35 + image[i][j+1]*0.35 + image[i-1][j]*0.15 + image[i-1][j+1]*0.15
                elif j == y-1:
                    new_image[i][j+1] = image[i][j]*0.5 + image[i+1][j]*0.25 + image[i-1][j]*0.25
                else:
                    new_image[i][j+1] = image[i][j]*0.3 + image[i][j+1]*0.3 + image[i+1][j]*0.1 + image[i-1][j]*0.1 + image[i+1][j+1]*0.1 + image[i-1][j+1]*0.1
                
    return new_image

"""
input: an image, and the number of seams to be added
output: same image but with seams added
"""
def add_columns(image,number):
    seams = np.empty((number,image.shape[0]))
    temp_image = image
    
    for h in range(number):
        sys.stdout.write("\r{0}%".format(int(h/(number*2)*100)+1))
        sys.stdout.flush()
        temp_image,seam = remove_column(temp_image)
        seams[h] = seam

    for h in range(number):
        sys.stdout.write("\r{0}%".format(int((h+number)/(number*2)*100)+1))
        sys.stdout.flush()
        seam = [x+h for x in seams[number-1-h]]
        image = add_column(image,seam)
        
    return image


        
name = input('Enter the name of the image: ')

image = cv2.imread(name,cv2.IMREAD_COLOR)
image = image.astype('float64')

width = image.shape[1]
height = image.shape[0]

print('Your image is currently of size: ' + str(width) + 'x' + str(height))
new_width = int(input('Enter your new width: '))
new_height = int(input('Enter your new height: '))
energy = input('Enter energy type [backward/forward]: ')
if energy == 'backward':
    FORWARD_ENERGY = 0
else:
    FORWARD_ENERGY = 1

if new_width < width:
    print('Removing pixels vertically:')
    image = remove_columns(image,width-new_width)
if new_height < height:
    print('Removing pixels horizontally:')
    image = rotate(image)
    image = remove_columns(image,height-new_height)
    image = rotate(image)
if new_width > width:
    print('Adding pixels vertically:')
    image = add_columns(image,new_width-width)
if new_height > height:
    print('Adding pixels horizontally:')
    image = rotate(image)
    image = add_columns(image,new_height-height)
    image = rotate(image)

cv2.imwrite('final_image.jpg', image)

print('\nDone')
