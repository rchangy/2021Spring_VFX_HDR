# 2021Spring_VFX_HDR

## Usage
pack all the images in one folder, and two text files are required ([sample](data/original)):
- imlist.txt: image file name and shutter time
- sel_pix.txt: the coordinate position of pixels chosen for response curve recovering

### Alignment(Optional):
```
python3 align.py [directory]
```

### Preprocessing
```
python3 pre_process.py [directory]
```
use argument `-align` if alignment is done 

### Color Separation
```
python3 color_ratio.py [directory]
```

### Response Curve Recovering
run the command in matlab
```
solve_g.m [directory]
```

### HDR
```
python3 radiance_map.py [directory]
```
use argument `-align` if alignment is done 

### Tone Mapping
```
python3 tone_mapping.py [directory] [method] [arguments]
```
methods and arguments:
- global operator: -global [alpha]
- adaptive logarithmic: -adaptive [b] [alpha]
- bilateral filtering: -bilateral [target compression] [sigmaR]
- gamma correction: -gamma [gamma]

## Result
[original images](data/original)</br>
[result](data/tone_mapped)

