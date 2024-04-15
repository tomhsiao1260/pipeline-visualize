<h1 align="center">Pipeline Visualize</h1>

## Introduction

Try to visualize the intermediate states in [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) pipeline. Currently for point cloud step only.

## Usage

Install dependency

```
pip install numpy torch tifffile scipy open3d
```

Choose a cell block at [here](http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volume_grids/20230205180739/) and then run the script. Here's an example.

```python
python surface_detection.py cell_yxz_006_008_004.tif
```

This is a script slightly modify from ThaumatoAnakalyptor point cloud step. It generates an `output` folder to save intermediate info as torch tensors. Checkout [here](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/documentation/ThaumatoAnakalyptor___Technical_Report_and_Roadmap.pdf) to learn more.

Let's visualize those tensors

```python
python visualize.py
```



## Future Notes

