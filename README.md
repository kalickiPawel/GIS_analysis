# GIS analysis

## Repository consist:
- GIS Interpolation class

## Interpolation
### TODOs

- [x] loading of bathymetric data from ASCII file
- [x] user input -> spacing of grid (horizontal and vertical distances between adjacent points on the grid)
- [x] user input -> size of window
- [x] user input -> type of window -> square or circle
- [x] user input -> minimum value of points to calculate new point (if count of points is lower set NaN)
- [ ] user input -> type of method the interpolation (ma, idw, kriging)
    - [ ] user input -> if method is IDW -> set power value
    - [ ] user input -> if method is Kriging -> set method of variogram
- [ ] implementation of interpolation with moving average method
    - [ ] for window as square
    - [x] for window as circle
- [ ] implementation of interpolation with IDW method
    - [ ] for window as square
    - [ ] for window as circle
- [ ] implementation of interpolation with Kriging method
    - [ ] for window as square
    - [ ] for window as circle
- [ ] calculate the size of grid -> !to_fix
- [x] interpolation with one of method
- [x] implementation of progress bar
- [x] display the surface by any method -> 2D
- [x] display the surface by any method -> 3D
- [x] save data to file with any format -> CSV
- [ ] save data to file with w formacie ASCII Grid XYZ format (more: https://gdal.org/drivers/raster/xyz.html)

### Improvements
- [ ] Add the color map to display

## Compression
### TODOs


## Development

1. Create virtual environment with `virtualenv .venv`.
2. Activate venv with `source .venv/bin/activate`.
3. Install packages with `pip install -r requirements.txt`.
4. Launch application with `python main.py`.
