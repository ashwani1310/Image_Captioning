# Image_Captioning
Classifying CIFAR-10 images 
The project uses “python” to develop the code. The deep learning library used in this project is “CNTK”. 

Requirements:

      1) Anaconda
      2)  Python 2.7
      3)  CNTK

We recommend you to  use Anaconda as it comes pre installed with Python and the other python modules like numpy, scipy, scikit-learn, matplotlib, etc.

Install Anaconda using the following link: https://docs.continuum.io/anaconda/install


For Linux Users:

CNTK requires OpenMPI 1.10.x to be installed on your system. On Ubuntu 16.04 install it by typing this in your terminal:
	sudo apt-get install openmpi-bin
After the above requirements are fulfilled, run the following command to install CNTK:
	Pip install <url>

Where url: 

For CPU only system:  https://cntk.ai/PythonWheel/CPU-Only/cntk-2.0rc2-cp27-cp27mu-linux_x86_64.whl

For GPU system:
https://cntk.ai/PythonWheel/GPU/cntk-2.0rc2-cp27-cp27mu-linux_x86_64.whl



For Windows Users:
After the above requirements are fulfilled, run the following command in your command prompt to install CNTK:
	C:\> pip install <url>


Where url: 

For CPU only system:  
https://cntk.ai/PythonWheel/CPU-Only/cntk-2.0rc2-cp27-cp27m-win_amd64.whl

For GPU system:
https://cntk.ai/PythonWheel/GPU/cntk-2.0rc2-cp27-cp27m-win_amd64.whl


Data Set:
The Training and testing data set used for training and validating the neural network is the CIFAR-10 dataset comprising of 50,000 images.
The data set contains image vectors with each vector representing input image features i.e. the different pixel values and a label vector labeling the 0 or 1 depending upon the image.

How to run Code:
Run the python script named “image_classify.py” .
The “image_classify.py” imports within it the “classifier.py” python script which is also present in the same folder.
