# Surface Enlargement Factor

This Python package provides functionality to read and process surface topography data
(list or matrix format), compute surface areas (linear or triangular), and optionally
remove bending from the data.

## Key Features
- Flexible input (list or matrix)
- Optional unit conversion
- Multiple surface area methods
- Bending correction (row-wise)
- Plotting
- Result writing to text files

## Installation
pip install .

## Usage
Check out examples/run_example.py for a simple demonstration.


## Surface Enlargement Factor (mathematical framework)

When surface topographies are recorded using confocal Raman microscopy or atomic force microscopy, the resulting data is arranged on a regular grid. A triangular mesh is then constructed over the grid to approximate the true surface area. The ratio between the calculated (true) surface area and the projected geometric area defines the **Surface Enlargement Factor (SEF)**, which quantifies the increase in area due to roughness.

Surface topography data is provided as a 2D array of height values z_{i,j} on a grid of coordinates (x_i,y_j) with uniform spacings delta_x and delta_y. For each grid cell defined by points  
(i,j), (i+1,j), (i,j+1), and (i+1,j+1), two triangles are constructed:

1. **First Triangle:**  
   Formed by the points  
   (x_i,y_j,z_{i,j}) (top-left),  
   (x_{i+1},y_j, z_{i+1,j}) (bottom-left), and  
   (x_{i+1},y_{j+1}, z_{i+1,j+1}) (bottom-right).

2. **Second Triangle:**  
   Formed by the points  
   (x_i, y_j, z_{i,j}) (top-left),  
   (x_{i+1}, y_{j+1}, z_{i+1,j+1}) (bottom-right), and  
   (x_i, y_{j+1}, z_{i,j+1}) (top-right).

<img src="/triangular_mesh_with_labels.png" alt="Exemplary mesh built from triangles" width="500"/>
*Figure: A 4×4 section of the grid as recorded with confocal Raman microscopy. The red lines indicate the first triangle and the yellow lines indicate the second triangle formed within the first grid cell.*

For each triangle with vertices A, B, and C, the side lengths are computed as

a = |B - A|,
b = |C - B|,
c = |A - C|.

The semiperimeter s is

s = (a+b+c)/2,

and the area of the triangle is given by Heron's formula:

A_{triangle} = sqrt(s(s - a)(s - b)(s - c)).

The 'true' surface area is obtained by summing the areas of all triangles:

A_{surface} = sum_{k=1}^{M} (A_{triangle_k}),

while the projected geometric area is

A_{{geometric} = (N_x - 1)(N_y - 1) delta_x delta_y.

Finally, the SEF is defined as

SEF = A_{surface}/A_{geometric}).

A finer grid (smaller delta_x and delta_y) improves the resolution of the mesh and the accuracy of the SEF, thereby capturing the nuances of surface roughness.
