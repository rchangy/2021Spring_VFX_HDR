import sys
import csv
import numpy as np
import cv2 as cv

file_path = sys.argv[1]
imlist_path = file_path+'/imlist.txt'

#read image
with open(imlist_path, "r") as imlist:
    P = int(imlist.readline())
    img_path = [0] * P
    for i in range(P):
        img_path[i] = imlist.readline().split()[0]
img = [cv.imread(file_path+'/'+img_path[i]) for i in range(P)]
m, n, chn = img[0].shape

#offset
offset = np.zeros((P, 2), dtype=int)

if len(sys.argv) > 2:
    align_path = imlist_path = file_path+'/offset.csv'
    with open(align_path, 'r') as align:
        rows = csv.reader(align)
        i = 0
        for row in rows:
            offset[i] = np.array(row, dtype=int)
            i += 1

color_ratio = np.zeros((m, n, 3))

w_color = np.array([0.114, 0.587, 0.299])
intensity = [0] * P
w = np.zeros(256)


for j in range(128):
    w[j] = j
for j in range(128, 256):
    w[j] = 256-j

for p in range(P):
    intensity[p] = np.array(np.dot(img[p], w_color), dtype='int')
for i in range(m):
    for j in range(n):
        W = 0
        for p in range(P):
            for chn in range(3):
                color_ratio[i, j, chn] += w[intensity[p][i, j]] * img[p][i, j, chn]
                W += w[intensity[p][i, j]]
        if W > 0:
            color_ratio[i, j] /= W



with open(file_path+'/color.csv', 'w') as col:
    writer = csv.writer(col)
    writer.writerows(color_ratio[:, :, 0])
    writer.writerows(color_ratio[:, :, 1])
    writer.writerows(color_ratio[:, :, 2])
print('B_mean: ',color_ratio[:,:,0].mean())
print('G_mean: ',color_ratio[:,:,1].mean())
print('R_mean: ',color_ratio[:,:,2].mean())