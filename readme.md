<h1 align="center">Pipeline Visualize</h1>

## Introduction

Try to visualize the intermediate states in [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) pipeline. Currently for point cloud step only.

## Usage

Install dependency

```
pip install numpy torch tifffile scipy open3d opencv-python
```

Choose a cell block [here](http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volume_grids/20230205180739/) and run the script. It's slightly modify from ThaumatoAnakalyptor point cloud step. And will generate an `output` folder to save intermediate info as torch tensors. Checkout [here](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/documentation/ThaumatoAnakalyptor___Technical_Report_and_Roadmap.pdf) to learn more. Here's an example command if you place the data in root folder.

```python
python surface_detection.py cell_yxz_006_008_004.tif
```

Now, let's visualize those tensors 🙌

```python
python visualize.py
```

## Future Works

I think it's a great starting point to learn ThaumatoAnakalyptor. Tweak values with visual feedback. Try to figure out which parts could be done better. We can go even further by putting those output videos on an endless whiteboard. Here's a [prototype demo](https://twitter.com/yao1260/status/1778078347299627202). This part has not been started yet, but may be implemented via [VC Whiteboard](https://github.com/tomhsiao1260/vc-whiteboard).

In addition, I will also discuss with VC community how to integrate these python scripts into ThaumatoAnakalyptor itself to avoid maintaining too much identical code.

Let's visualize the entire pipeline 🙌

