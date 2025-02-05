# Microflow-IOP
## TLDR
MicroFlow-IOP is a computer vision-based algorithm designed to analyze intraocular pressure (IOP) by tracking liquid displacement within a microfluidic chamber. Utilizing optical flow techniques, particularly the Lucas-Kanade method, the algorithm accurately follows the movement of the liquid surface as it responds to pressure changes. By integrating real-time tracking, noise reduction, and robust feature selection, MicroFlow-IOP ensures precise measurement of fluid displacement, which is then mapped to IOP values using a calibrated model.

![displacement_demo](displacement_demo.png)

This system enhances the reliability of non-invasive IOP monitoring by addressing challenges such as occlusions, reflections, and color-based interference in microfluidic systems. With potential applications in ophthalmology and biomedical research, MicroFlow-IOP represents a step toward more accessible and automated eye health diagnostics.

## Technical Details
### Liquid surface tracking
Due to the relative small cross section of the microfluidic chamber and goal of using accessible consumer-grade accessories (e.g. webcam, phone camera) which usually lack super-high resolution, **Lucas Kanade** (LK) method is used. LK method is a optical flow tracking algorithm is based on the assumption of relative intensity constancy, and infer the location of point-of-interest based on intensity change in specified pixels. Technical theorem can refer to [Lucas-Kanade Optical Flow](https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf).

## Deployment
1. Change to directory of your project. Create virtual environment and load the dependencies according to `IOP_requirements.txt`.
```
python -m venv [name of virtual environment]
source [name of virtual environment]/bin/activate
pip install -r ./IOP_requirements.txt
```
2. Run the main.py file in the repository
```
python main.py
```