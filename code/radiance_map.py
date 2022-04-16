import sys
import csv
import numpy as np
import cv2 as cv

file_path = sys.argv[1]
imlist_path = file_path+'/imlist.txt'
g_path =  file_path+'/g.csv'

#read in g.csv
g = np.zeros((3, 256), dtype='double')
w = np.zeros(256, dtype='double')
with open(g_path, 'r') as g_file:
    rows = csv.reader(g_file)
    i = 0
    for row in rows:
        if i < 3:
            g[i] = np.asarray(row, dtype='double')
            i += 1
        else:
            w = np.asarray(row, dtype='int')
for j in range(127, 256):
    w[j] += 1
#read image
with open(imlist_path, "r") as imlist:
    P = int(imlist.readline())
    img_path = [0] * P
    SS = np.zeros(P)
    for i in range(P):
        img_path[i], SS[i] = imlist.readline().split()
        SS[i] = float(SS[i])

img = [cv.imread(file_path+'/'+img_path[i]) for i in range(P)]
m, n, chn = img[0].shape
SS = np.log(SS)  #delta t

#offset
offset = np.zeros((P, 2), dtype=int)

if len(sys.argv) > 2:
    align_path = imlist_path = file_path+'/offset.csv'
    with open(align_path, 'r') as align:
        rows = csv.reader(align)
        i = 0
        for row in rows:
            offset[i] = np.array(row, dtype=float)
            offset[i] = np.array(offset[i], dtype=int)
            i += 1


r_map = np.zeros_like(img[0], dtype='double')
lnE = np.zeros_like(img[0], dtype='double')

for i in range(m):
    for j in range(n):
        E = np.zeros(3)
        W = np.zeros(3)
        for p in range(P):
            for chn in range(3):
                try:
                    E[chn] += w[img[p][i + offset[p][0], j + offset[p][1], chn]] * (g[chn][img[p][i + offset[p][0], j + offset[p][1], chn]] - SS[p])
                    W[chn] += w[img[p][i + offset[p][0], j + offset[p][1], chn]]
                except:
                    E[chn] += 0
                    W[chn] += 0
        for q in range(3):
            if W[q] > 0:
                lnE[i, j, q] = E[q] / W[q]
                r_map[i, j, q] = np.exp(lnE[i, j, q])
            else:
                lnE[i, j, q] = -float('inf')
                r_map[i, j, q] = 0
        
output_path = file_path+'/raw.hdr'
cv.imwrite(file_path+"/out.hdr", r_map)

with open(output_path, 'w') as out:
    writer = csv.writer(out)
    writer.writerow([m, n])
    writer.writerows(lnE[:, :, 0])
    writer.writerows(lnE[:, :, 1])
    writer.writerows(lnE[:, :, 2])
