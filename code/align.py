import cv2 as cv
import numpy as np
import sys
import csv
def gaussian_filter(img, p_x, p_y):
    W = 0
    sigma = 2
    ret = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            w = np.exp(-((p_x-x)**2 + (p_y-y)**2) / 2*sigma**2)
            W += w
            ret += w * img[x, y]
    if W > 0:
        ret = ret/W
    return ret
            

class Pyramid:
    def __init__(self, img, max_offset = 8):
        self.pyramids = list()
        self.exclusion = list()
        self.level = int(np.log2(max_offset))
        size = 5
        bound = int(size/2)
        c = 0
        for im in img:
            p_img = list()
            p_img.append(im)
            m, n = im.shape
            for i in range(1,self.level):
                last_m = m
                last_n = n
                m = int(m/2)
                n = int(n/2)
                p = np.zeros((m, n))
                for x in range(m):
                    for y in range(n):
                        p[x, y] = gaussian_filter(p_img[-1][max(2*x-bound, 0):min(2*x+bound+1, last_m), max(2*y-bound, 0):min(2*y+bound+1, last_n)], 2*x - max(2*x-bound, 0), 2*y - max(2*y-bound, 0))
                p_img.append(p)
                
            #create bitmap
            p_bitmap = list()
            ex_bitmap = list()
            thres = 4
            for im in p_img:
                p_bool = np.zeros_like(im, dtype=bool)
                ex = np.zeros_like(im, dtype=bool)
                med = np.median(im)
                for x in range(im.shape[0]):
                    for y in range(im.shape[1]):
                        if im[x, y] >= med:
                            p_bool[x, y] = True
                        else:
                            p_bool[x, y] = False
                        if abs(im[x, y] - med) <= thres:
                            ex[x, y] = False
                        else:
                            ex[x, y] = True
                p_bitmap.append(p_bool)
                ex_bitmap.append(ex)
            self.pyramids.append(p_bitmap)
            self.exclusion.append(ex_bitmap)
            print('image ' + str(c) + ' shrinked')
            c += 1

        self.P = c
        return
    def align(self, std = 0):
        self.offset = np.zeros((self.P, 2))
        standard = self.pyramids[std]
        for p in range(self.P):
            p_offset = np.array([0, 0])
            if p == std:
                print("standard image " + str(p))
                continue
            for i in range(self.level-1, -1, -1):
                p_offset *= 2
                m, n = self.pyramids[p][i].shape
                min_err = m*n
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        shifted_bitmap = np.zeros_like(self.pyramids[p][i])
                        shifted_ex = np.zeros_like(self.pyramids[p][i])
                        shifted_bitmap[max(0, -(p_offset[0]+x)):min(m, m-(p_offset[0]+x)), max(0, -(p_offset[1]+y)):min(n, n-(p_offset[1]+y))] = self.pyramids[p][i][max(0, p_offset[0]+x):min(m, m+(p_offset[0]+x)), max(0, p_offset[1]+y):min(n, n+(p_offset[1]+y))]
                        shifted_ex[max(0, -(p_offset[0]+x)):min(m, m-(p_offset[0]+x)), max(0, -(p_offset[1]+y)):min(n, n-(p_offset[1]+y))] = self.exclusion[p][i][max(0, p_offset[0]+x):min(m, m+(p_offset[0]+x)), max(0, p_offset[1]+y):min(n, n+(p_offset[1]+y))]
                        diff = np.logical_xor(self.pyramids[std][i], shifted_bitmap)
                        diff = np.logical_and(diff, self.exclusion[std][i])
                        diff = np.logical_and(diff, shifted_ex)
                        err = np.sum(diff)
                        #print('pic', p, 'level', i, 'err:', err)
                        if err < min_err:
                            min_err = err
                            min_x = x
                            min_y = y
                        
                p_offset[0] += min_x
                p_offset[1] += min_y
            self.offset[p] = p_offset
            print('image '+str(p)+' offset computed:', p_offset)
        return
            



file_path = sys.argv[1]
imlist_path = file_path +'/imlist.txt'

with open(imlist_path, "r") as imlist:
    P = int(imlist.readline())
    img_path = [0] * P
    for i in range(P):
        img_path[i]= imlist.readline().split()[0]
        print(i, img_path[i])
img = [cv.imread(file_path +'/'+ img_path[i], cv.IMREAD_GRAYSCALE) for i in range(P)]

p = Pyramid(img)
p.align()
with open(file_path +'/offset.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerows(p.offset)
