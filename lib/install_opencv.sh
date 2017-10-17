#!/usr/bin/env bash

# INSTALL THE LIBRARY (YOU CAN CHANGE '3.2.0' FOR THE LAST STABLE VERSION)
OPENCV_VERSION=3.2.0
OPENCV_TARGET_DIR=$(realpath opencv-${OPENCV_VERSION})
OPENCV_CONTRIB_TARGET_DIR=$(realpath opencv_contrib-${OPENCV_VERSION})

if ! [[ -e ${OPENCV_VERSION}.tar.gz ]]
then  
	echo "Downloading OpenCV"
	wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz
else 
	echo "OpenCV was already downloaded..."
fi

if ! [[ -e ${OPENCV_VERSION}_contrib.tar.gz ]]
then  
	echo "Downloading OpenCV_contrib"
	wget -O ${OPENCV_VERSION}_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz
	
else 
	echo "OpenCV_contrib was already downloaded..."
fi



if [[ -d ${OPENCV_TARGET_DIR} || -d ${OPENCV_CONTRIB_TARGET_DIR} ]]
then  
	while ! [[ $key == "y" || $key == "n" ]]
	do	
		read -n1 -r -p "Directory OpenCV or OpenCV_contrib already exist. Shall we remove it before continuing? (y/n)" key
		echo ""
	done
	if  [[ $key == "y" ]] 
	then  
		echo "Deleting existing directory..."
		rm -rfv ${OPENCV_TARGET_DIR}
		rm -rfv ${OPENCV_CONTRIB_TARGET_DIR}
	else 
		echo "Sorry, can't continue with an dirty directory..." 
		exit 1
	fi
fi

tar xf ${OPENCV_VERSION}_contrib.tar.gz
tar xf ${OPENCV_VERSION}.tar.gz
cd ${OPENCV_TARGET_DIR}
mkdir build
cd build
cmake -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF \
                      -DWITH_CUDA=OFF \
                      -DWITH_FFMPEG=OFF \
                      -DENABLE_AVX=ON \
                      -DWITH_OPENGL=ON \
                      -DWITH_OPENCL=ON \
                      -DWITH_IPP=ON \
                      -DWITH_TBB=ON \
                      -DWITH_EIGEN=ON \
                      -DWITH_V4L=ON \
                      -DBUILD_TESTS=OFF \
                      -DBUILD_PERF_TESTS=OFF \
                      -DCMAKE_BUILD_TYPE=RELEASE \
                      -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
		      -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_TARGET_DIR}/modules \
                      -DPYTHON3_EXECUTABLE=$(which python) \
                      ..
# DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") 
# -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") 

echo "cmake finished his job."
read -n1 -r -p "Press any key to continue..." key

# cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig


# EXECUTE SOME OPENCV EXAMPLES AND COMPILE A DEMONSTRATION

# To complete this step, please visit 'http://milq.github.io/install-opencv-ubuntu-debian'.

