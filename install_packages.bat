@echo off
rem Install required Python packages using pip
pip install Flask-SQLAlchemy
pip install PyMySQL
pip install torch torchvision
pip install torchxrayvision
pip install tabulate
pip install grad-cam
pip install scikit-learn
pip install scikit-image
pip install pandas
pip install numpy
pip install matplotlib
pip install opencv-python
pip install pydicom
pip install Pillow

rem Run this script again to ensure all packages are installed
echo.
echo All packages installed. Press any key to exit.
pause > nul
