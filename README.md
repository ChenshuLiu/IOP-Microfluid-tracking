# Microflow-IOP
## TLDR
MicroFlow-IOP is a computer vision-based algorithm designed to analyze intraocular pressure (IOP) by tracking liquid displacement within a microfluidic chamber. Utilizing optical flow techniques, particularly the Lucas-Kanade method, the algorithm accurately follows the movement of the liquid surface as it responds to pressure changes. By integrating real-time tracking, noise reduction, and robust feature selection, MicroFlow-IOP ensures precise measurement of fluid displacement, which is then mapped to IOP values using a calibrated model.

![displacement_demo](displacement_demo.png)

This system enhances the reliability of non-invasive IOP monitoring by addressing challenges such as occlusions, reflections, and color-based interference in microfluidic systems. With potential applications in ophthalmology and biomedical research, MicroFlow-IOP represents a step toward more accessible and automated eye health diagnostics.

## Technical Details

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