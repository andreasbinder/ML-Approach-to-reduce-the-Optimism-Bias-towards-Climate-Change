# Important

The original repository can be found [here](https://github.com/andreasbinder/AppliedML_using_GAN).
In case of in problems, please check out the repo on github.
Concerning functionality, there should be no difference.

# ML-Approach to reduce the Optimism Bias towards Climate Change
 
This project is part of a seminar at the Technical University Munich focusing on sustainable development.
It's goal is to build a Generative Adversarial Network being able to generate pictures of burning houses and deploy it to the web via conversions supplied by tensorflowjs.
The running prototype can be found [here](http://home.in.tum.de/~bindera). 
 
## Getting started
Here's what you need to set up the project.
### Prerequisites
Since this project is written in Python you will need to have a Python version >= 3.6.8 running on your system.
Other you can upgrade it as follows:
```bash
sudo apt-get install python3
```
You can get the package manager by typing:
```bash
sudo apt-get install python3-pip
# you might need to update the package manager to activate the package
sudo apt-get update
```

The deep learning framework used is Keras with a Tensorflow backend. In case any problems arise, please check 
out the official documentation for Keras [here](https://keras.io/#installation) and for Tensorflow [here](https://www.tensorflow.org/install/pip?lang=python3).

```bash
# Keras version used 2.1.1
pip3 install Keras
# Tensorflow version used 1.15.0
# Please be aware that tensorflow 2.0 might be not compatible
pip install --upgrade tensorflow==1.15
```
For image processing we make use of the PIL library. Please get the newest version accordingly.
```bash
pip3 install Pillow
```
If you wanted to deploy your own prototype to a Website, you would need to convert it to a Tensorflow.js model.
The script is placed in the folder Scripts/ConvertModel.ssh. To execute the script, you need to have the tensorflowjs_converter.
By installing installing tensorflowjs you will be provided with the converter.
```bash
pip3 install tensorflowjs
```
For more detailed instructions, please check out their [homepage](https://www.tensorflow.org/js/tutorials/conversion/import_keras).
### Installation
You can install the project by cloning it via https:
```bash
git clone https://github.com/andreasbinder/AppliedML_using_GAN.git
```
I have coded the implementation on Linux Mint 19.2 . Hence, you might need to adjust the relative paths, especially if you are a Windows user.
## Test
You can run a demo of it by executing this code in the commandline when navigated into the project:
```python
python3 prototype/Test/InstallationTest/InstallationTest.py
```
After few seconds you should see some generated pictures in this folder Test/InstallationTest/Results .

To experiment with the code yourself, please check out the Model directory.

## Author
Andreas Josef Binder, B.Sc. Information Systems, Technical University Munich

# ML-Approach-to-reduce-the-Optimism-Bias-towards-Climate-Change
