# CSE 591 Perception in Robotics

## Environment Setup

Using python 3 install the following:

```
pip install matplotlib
pip install numpy
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```

## Run the code

```
python main.py  # this needs to be python 3
```

depth(x) = fB/(x - x')
f: fundamental matrix f = K^(-t)EK'^(-1)
E: essential matrix E = t x R
B: basline

(Calibration explanation)[https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file-]
