import sys
import numpy as np
import cv2 as cv
import csv
from scipy.signal import convolve2d
from scipy.interpolate import griddata

class tone_mapping:
    def __init__(self, data, color_r):
        self.lE = data
        self.E = np.exp(lE)
        self.m = data.shape[0]
        self.n = data.shape[1]
        #compute intensity
        w = np.array([0.114, 0.587, 0.299])
        self.intensity = np.dot(self.E, w)

        #compute color ratio
        color_intense = np.dot(color_r, w)
        self.color_ratio = np.zeros_like(self.E)
        self.color_ratio[:,:,0] = color_r[:,:,0] / color_intense
        self.color_ratio[:,:,1] = color_r[:,:,1] / color_intense
        self.color_ratio[:,:,2] = color_r[:,:,2] / color_intense
        self.mapped = self.intensity
        return
    def coloring(self):
        
        self.result = np.zeros_like(self.E)
        for chn in range(3):
            self.result[:, :, chn]  = self.mapped * self.color_ratio[:, :, chn]
        return self.result

    def global_operator(self, key = 0.18):
        print('global operator')
        print('alpha: ', key)
        self.mapped = np.zeros_like(self.intensity)
        log_intensity = np.log(self.intensity + np.finfo(np.float64).tiny)
        Lw = np.exp(log_intensity.sum() / (self.m * self.n))
        k = key/Lw
        self.mapped = k * self.intensity
        L_white = self.mapped.max()
        self.mapped = self.mapped * (1+(self.mapped / L_white**2)) / (1+self.mapped)
        self.mapped = self.mapped * 255
        return
    def adaptive_log(self, b = 0.85, key = 0.18):
        print('adaptive logarithmic')
        print('b: ', b)
        print('alpha: ', key)
        self.mapped = np.zeros_like(self.intensity)
        log_intensity = np.log10(self.intensity + np.finfo(np.float64).tiny)
        L_wa = np.exp(log_intensity.sum() / (self.m * self.n))
        scaled_intensity = self.intensity * key / L_wa
        L_wmax = scaled_intensity.max() * (1 + b - 0.85)**5
        w = 255 * 0.01 / np.log10(L_wmax + 1)
        b = np.log(b) / np.log(0.5)
        for i in range(self.m):
            for j in range(self.n):
                    bias = (scaled_intensity[i, j] / L_wmax)**b
                    self.mapped[i, j] = w * np.log(scaled_intensity[i, j] + 1) / np.log(2 + bias * 8)
        max_intensity = self.mapped.max()
        self.mapped *= 255 / max_intensity

        return

    def gamma_correction(self, g=2.2):
        print('gamma correction')
        print('gamma: ', g)
        intensity_max = self.mapped.max()
        self.mapped = ((self.mapped / intensity_max) ** (1/g)) * 255
        return
    def bilateral(self, compression = 5, sigmaR = 0.5):
        print('bilateral filtering')
        print('target compression: ', compression)
        print('sigmaR: ', sigmaR)
        sigmaS = 64
        sampling = 32
        log_intensity = np.log10(self.intensity + np.finfo(np.float64).tiny)
        maxI = np.max(log_intensity)
        minI = np.min(log_intensity)
        dynamic = maxI-minI
        #bilateral
        base = np.zeros_like(self.intensity)
        #image downsampling
        down_m = round(self.m / sampling) + 1
        down_n = round(self.n / sampling) + 1
        downsample = np.zeros((down_m, down_n))
        down_sigmaS = sigmaS / sampling
        count = np.zeros((down_m, down_n))
        for x in range(self.m):
            for y in range(self.n):
                downsample[round(x/sampling), round(y/sampling)] += float(log_intensity[x, y])
                count[round(x/sampling), round(y/sampling)] += 1
        downsample = downsample / count
        #kernel
        kernel_half_size = int(sigmaS/sampling)
        kernel_size = 2 * kernel_half_size + 1
        padding = kernel_size - 1
        kernel = np.zeros((kernel_size, kernel_size))
        for x in range(-kernel_half_size, kernel_half_size+1):
            for y in range(-kernel_half_size, kernel_half_size+1):
                kernel[kernel_half_size+x, kernel_half_size+y] = np.exp(-(x**2 + y**2) / (2 * down_sigmaS**2))
        #compute j
        segments = round(dynamic / sigmaR)

        for j in range(segments+1):
            i = minI + j * dynamic / segments
            G = np.zeros((down_m+2*padding, down_n+2*padding))
            G[padding:G.shape[0] - padding, padding:G.shape[1]-padding] = np.exp(-(downsample - i)**2 / (2*sigmaR**2))
            K = convolve2d(G, kernel, mode='same')
            K = np.where(K == 0, 1, K)
            H = np.zeros_like(G)
            H[padding:H.shape[0]-padding, padding:H.shape[1]-padding] = G[padding:G.shape[0] - padding, padding:G.shape[1]-padding] * downsample
            H = convolve2d(H, kernel, mode='same')
            J = H/K
            #upsampling interpolation
            points = np.zeros((J.size, 2))
            points[:, 0] = np.repeat(np.array(np.meshgrid(range(J.shape[0]))), J.shape[1]).flatten()
            points[:, 1] = np.array(np.meshgrid(range(J.shape[1]))*J.shape[0]).flatten()
            interp_points = np.zeros((self.m*self.n, 2))
            interp_points[:,0] = (np.repeat(np.array(np.meshgrid(range(self.m))), self.n) / sampling + padding).flatten()
            interp_points[:,1] = (np.array(np.meshgrid(range(self.n)) * self.m) / sampling + padding).flatten()
            interp_value = griddata(points, J.flatten(), interp_points)
            #interpolation weight
            interpolation_weight = np.ones((m, n)) - (abs(log_intensity - i) / (dynamic / segments))
            interpolation_weight = np.where(interpolation_weight < 0, 0, interpolation_weight)
            base += interp_value.reshape((self.m , self.n)) * interpolation_weight
        detail = log_intensity - base
        compression_factor = np.log10(compression) / (np.max(base) - np.min(base))
        out_intensity = log_intensity*compression_factor - np.max(base) * compression_factor
        self.mapped = 10 ** (out_intensity)
        max_intensity = self.mapped.max()
        self.mapped *= 255 / max_intensity
        return



raw_path = sys.argv[1] + '/raw.hdr'
with open(raw_path, 'r') as raw:
    rows = csv.reader(raw)
    flag = 0
    i = 0
    for row in rows:
        if flag == 0:
            m = int(row[0])
            n = int(row[1])
            lE = np.zeros((m, n, 3))
            flag = 1
        else:
            lE[i%m, :, int(i/m)] = np.array(row, dtype='double')
            i += 1

with open(sys.argv[1] + '/color.csv', 'r') as col:
    rows = csv.reader(col)
    color_r = np.zeros((m, n, 3))
    i = 0
    for row in rows:
            color_r[i%m, :, int(i/m)] = np.array(row, dtype='double')
            i += 1
tm = tone_mapping(lE, color_r)
flag = 1
color = 1
outfile_path = sys.argv[1] + "/tone_mapped.png"

for i in range(2, len(sys.argv)):
    if sys.argv[i] == '-global':
        tm.global_operator(key = float(sys.argv[i+1]))
    if sys.argv[i] == '-adaptive':
        tm.adaptive_log(b = float(sys.argv[i+1]), key=float(sys.argv[i+2]))
    if sys.argv[i] == '-bilateral':
        tm.bilateral(compression=float(sys.argv[i+1]), sigmaR=float(sys.argv[i+2]))
    if sys.argv[i] == '-gamma':
        tm.gamma_correction(g=float(sys.argv[i+1]))

cv.imwrite(sys.argv[1]+'/intensity.png', tm.mapped)
result = tm.coloring()
cv.imwrite(outfile_path, result)
