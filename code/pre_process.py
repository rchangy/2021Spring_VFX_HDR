import sys
import numpy as np
import csv
import cv2 as cv

# read in images
file_path = sys.argv[1]
imlist_path = file_path+'/imlist.txt'


with open(imlist_path, "r") as imlist:
    P = int(imlist.readline())
    img_path = [0] * P
    SS = np.zeros(P)
    for i in range(P):
        img_path[i], SS[i] = imlist.readline().split()
        SS[i] = float(SS[i])
        print(i, img_path[i], SS[i])

img = [0] * P
for i in range(P):
    img[i]= cv.imread(file_path+'/'+img_path[i])
m, n, chn = img[0].shape

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

#get pixel values
sel_pix_path = file_path+'/sel_pix.txt'
with open(sel_pix_path, "r") as sel_pix:
    N = int(sel_pix.readline())
    Z = np.zeros((N, P, 3))   #(chn, pix_id, img)
    for i in range(N):
        x, y = map(int, sel_pix.readline().split())
        for j in range(P):
            try:
                Z[i, j, :] = img[j][x + offset[j][0], y + offset[j][1], :]
            except:
                Z[i, j, :] = np.array([0, 0, 0])
            

#write pixel value and t to file
pix_out_path = file_path +'/pixels.csv'
with open(pix_out_path, "w") as pix_out:
    writer = csv.writer(pix_out)
    writer.writerows(Z[:, :, 0].astype("uint8"))
    writer.writerows(Z[:, :, 1].astype("uint8"))
    writer.writerows(Z[:, :, 2].astype("uint8"))

SS_out_path = file_path +'/shutter.csv'
with open(SS_out_path, "w") as SS_out:
    writer = csv.writer(SS_out)
    writer.writerow(np.log(SS))
