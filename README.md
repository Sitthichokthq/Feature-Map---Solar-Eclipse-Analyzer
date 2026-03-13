Feature Map & Eclipse Analyzer
+ CMD Plot  (Color-Magnitude Diagram)
Developer: Witchapon Puangkaew   |   Python 3 · OpenCV · Matplotlib · tkinter

📋  Overview
⚙️  Requirements
Python
Python >= 3.9

Dependencies (pip install)
pip install opencv-python numpy Pillow matplotlib
pip install scipy scikit-image

Note: scikit-image is optional — if not installed, the program still works but disables RANSAC circle fitting.

🚀  Installation & Run
# 1. clone / download
git@github.com:Sitthichokthq/Feature-Map---Solar-Eclipse-Analyzer.git
cd eclipse-analyzer

# 2. install dependencies
pip install -r requirements.txt

# 3. run
python eclipse_analyzer_with_cmd.py

🖥️  UI Layout
The main window is divided into 3 main sections:

Zone	Content	Proportion
Left	Original Image canvas	3
Center	Filtered Image canvas (+ star markers)	3
Top right	Analysis text box	2
Bottom right	Tab: Profiles | CMD Plot	2

🔧  Toolbar Reference

Button	Color	Function
Open Image	Blue	Load general images (PNG/JPG/BMP/TIFF)
Open Eclipse	Blue	Load eclipse images (multi-select → stack)
Filter (Combobox)	—	Select filter: Cross-Correlation / Sobel / Laplacian / Sharpen
Measure	Grey	Measure pixel distance on Filtered canvas (drag)
ROI	Grey	Select Region of Interest
Inspect	Grey	Click to view pixel intensity
Pan	Grey	Drag canvas to move image
Detect Circles	Green	Auto HoughCircles → draw outer/inner ring
Sobel Fit	Grey	Sobel edge + Taubin circle fit
Analyze Eclipse	Yellow	Full eclipse analysis
u (slider)	—	Limb darkening coefficient (0–1)
σ (spinbox)	—	Threshold sigma for star detection
ap (spinbox)	—	Aperture radius (pixel) for photometry
▶ Detect Stars	Purple	Run Aperture Photometry on Filtered image → CMD
Clear	Red	Clear all results

📖  Workflow
A)  General Image Analysis
1.	Click Open Image → select image file
2.	Choose desired Filter from Combobox
3.	Use Measure / ROI / Inspect tools as needed
4.	Click Detect Circles or Sobel Fit to find circles

B)  Eclipse Analysis
5.	Click Open Eclipse → select one or multiple files (auto stack)
6.	Adjust u slider (default 0.80 = Sun limb darkening)
7.	Click Analyze Eclipse → auto-detect Moon limb + Umbra edge
8.	View results in Analysis panel and Profiles tab (Outer/Inner Edge, Radial, Histogram)

C)  CMD Plot (Color-Magnitude Diagram)
9.	Load any image via Open Image
10.	Select Filter — CMD graph uses the currently displayed Filtered image
11.	Adjust σ (threshold) and ap (aperture radius) in the toolbar
12.	Click ▶ Detect Stars → runs Aperture Photometry + creates CMD
13.	View graph in CMD Plot tab (switchable with Profiles tab)
14.	Adjust Tip mT and Style (Dark/White), then click Set to redraw
15.	Click Save PNG to export the graph

📊  CMD Plot — 3 Panels

Panel	X axis	Y axis	Description
Left — Scatter CMD	Concentration Index (CI)	Instrumental Mag	Main scatter plot — color by kernel density (inferno/Blues)
Mid — CI Profile	CI (median ± 1σ)	Mag bin	Red line = median CI per magnitude bin
Right — Lum. Function	N stars	Mag	Luminosity function (binned + smoothed)

Note: Concentration Index (CI) = flux in small circle (r/2) / flux in large circle (r) — real stars have high CI (concentrated light), background/noise is low

🔬  Algorithms
Star Detection Pipeline
•	Grid background estimation (SExtractor-style, bicubic interpolate)
•	Local maximum filter + sigma-clip threshold
•	Aperture photometry: net flux = Σ(AP) − sky_median × n_AP
•	Concentration Index: CI = flux(r/2) / flux(r)
•	Instrumental magnitude: m = 25 − 2.5 × log₁₀(flux)

Eclipse Ring Detection
•	HoughCircles → Moon center & radius
•	Radial brightness profiles every 0.5° (720 rays)
•	Limb darkening correction: I(r) = I_obs / [1 − u(1 − cosθ)]
•	Savitzky-Golay smoothing + gradient sub-pixel interpolation
•	RANSAC circle fitting (scikit-image, optional)
•	Taubin algebraic circle fitting (fallback)
•	Constrained least-squares fit vs NASA theoretical radius

Image Stacking (multi-frame Eclipse)
•	HoughCircles per frame → scale & align all frames to reference
•	Median combine → reduce noise, cosmic rays

📐  Eclipse Output Fields

Field	Unit	Description
km/pixel	km/px	Scale from Moon diameter
Outer r	px / km	Measured Umbra radius
Outer area	km²	Umbra area
Inner r	px / km	Moon limb radius
Theoretical r	km	NASA theoretical Umbra radius
Radius error	%	Deviation |measured − theory| / theory
Area error	%	Area deviation
Shadow cover	%	Proportion of dark pixels in Moon disk
RANSAC	on/off	scikit-image RANSAC status

🌍  Physical Constants Used

Constant	Value	Unit
Earth radius (Rₑ)	6,371	km
Sun radius (R☉)	696,340	km
Sun-Earth distance (D☉)	149,597,870.7	km
Moon-Earth distance (D☽)	384,400	km
Moon diameter	3,474.8	km

🖱️  Mouse & Keyboard

Action	Result
Scroll wheel	Zoom in / out (0.05× – 40×)
Drag (Pan mode)	Move image
Drag (Measure mode)	Measure distance in pixels + km (if eclipse data)
Drag (ROI mode)	Select Region of Interest + show statistics
Click (Inspect mode)	Show intensity + distance from Moon center
Hover (CMD tab)	Show CI and mag at cursor in status bar

📁  Supported File Formats
•	PNG, JPG/JPEG, BMP
•	TIFF (8-bit and 16-bit)
•	CSV (for importing photometry catalogue: col1=mag_I, col2=mag_V)

💡  Tips & Troubleshooting
Stars not detected (Detect Stars)
•	Lower σ threshold (default 4.0 → try 2.0–3.0)
•	Increase aperture (ap) larger than star FWHM in image
•	Try different Filters — Sobel Edge often gives sharper star edges
Eclipse circles incorrect
•	Adjust Limb u to match actual limb darkening (typical 0.6–0.9)
•	Use Open Eclipse instead of Open Image for better Hough parameters
•	Multi-frame: select all files at once in dialog → auto stack
CMD looks abnormal
•	Concentration Index (CI) should be 0–1; if skewed, aperture is too small
•	Images with high background gradient: try Cross-Correlation filter before Detect Stars
•	Switch Style → White for export in academic documents

📜  License & Credit
This program is developed for educational and astronomical research purposes only. Commercial use is prohibited without permission.

Developer:  Sitthichokthq

Library	License	Used for
OpenCV	Apache 2.0	Image processing, HoughCircles, Sobel
NumPy	BSD	Array computation
Matplotlib	PSF	Plotting (CMD, profiles)
SciPy	BSD	Savgol filter, interpolation, optimization
Pillow	HPND	Image I/O, canvas rendering
scikit-image	BSD (opt.)	RANSAC circle fitting
tkinter	PSF	GUI

Feature Map & Eclipse Analyzer  +  CMD Plot

