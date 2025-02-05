# Microflow-IOP
## TLDR
MicroFlow-IOP is a computer vision-based algorithm designed to analyze intraocular pressure (IOP) by tracking liquid displacement within a microfluidic chamber. Utilizing optical flow techniques, particularly the Lucas-Kanade method, the algorithm accurately follows the movement of the liquid surface as it responds to pressure changes. By integrating real-time tracking, noise reduction, and robust feature selection, MicroFlow-IOP ensures precise measurement of fluid displacement, which is then mapped to IOP values using a calibrated model.

![displacement_demo](displacement_demo.png)

This system enhances the reliability of non-invasive IOP monitoring by addressing challenges such as occlusions, reflections, and color-based interference in microfluidic systems. With potential applications in ophthalmology and biomedical research, MicroFlow-IOP represents a step toward more accessible and automated eye health diagnostics.

## Technical Details
### Modeling IOP and distance
Based on experimental data (please refer to `distance_pressure.xlsx`), the pearson correlation coefficient is around 0.99, which is confident to say there is a linear relationship between distance and IOP.

### Liquid surface tracking
Due to the relative small cross section of the microfluidic chamber and goal of using accessible consumer-grade accessories (e.g. webcam, phone camera) which usually lack super-high resolution, **Lucas Kanade** (LK) method is used. LK method is a optical flow tracking algorithm is based on the assumption of relative intensity constancy, and infer the location of point-of-interest based on intensity change in specified pixels. Technical theorem can refer to [Lucas-Kanade Optical Flow](https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf).

![](https://viso.ai/wp-content/uploads/2021/03/optical-flow-opencv.jpg)

### Distance measurement
The liquid travels in a curved path in the microfluidic chamber, and the linear model constructed to measure IOP from distance is based on the scalar distance that the liquid surface travels. The distance travelled is being calculated incrementally through euclidean distance of new point tracked by Lucas-Kanade method from the last point location.

![](https://tutorial.math.lamar.edu/classes/calcii/ArcLength_Files/image001.gif)

### Direction of liquid surface travel
Because the distance is being determined incrementally, the direction of change matters. The liquid surface travelling direction is determined through dot product as: $$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos\theta$$ where when the point is moving in the positive directions the product is greater than zero, while negative directions cause the product to be smaller than zero.

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