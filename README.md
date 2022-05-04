<!-- <p align="center"> -->
  <!-- <img src="images/Logo.png" alt="Logo" width="68px" height="90px"> -->
<!-- </p> -->
<h1 align="left"> CDTL: Compatible-domain Transfer Learning for Breast Cancer Classification with Limited Annotated Data </h1>

<!-- <p align="center">  -->
<!-- <img src="images/histopathology.gif" alt="Animated gif" width="40%" height="40%"> -->
<!-- </p> -->

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>
<img src="images/Logo3.png" alt="Logo" align="right" width="31%" height="31%">
<!-- <img src="images/histopathology.gif" align="left" width="300px" height="230px"> -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview"> ➤ Overview</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#dataset"> ➤ Dataset </a></li>
    <li><a href="#license"> ➤ License </a></li>
    <li><a href="#reference"> ➤ Reference </a></li>
  </ol>
</details>


<!-- OVERVIEW -->
<h2 id="overview"> Overview</h2>

<p align="justify"> 
The aim of this study is to design a transfer learning framework to classify breast cancer cytological images into two categories: benign and malignant. Taking into account the ineffectiveness of employing natural images in TL to solve biomedical-domain problems, we propose the idea of Compatible-domain Transfer Learning (CDTL). This means that instead of using natural images (i.e. ImageNet) that are not essentially compatible with medical data, the pre-training phase of the model is performed employing histopathological images. We then fine-tune pre-trained models on the target data set containing limited cytological images).
</p>

<p align="center"> 
  <img src="images/finetuning.png" alt="finetuning" width="60%" height="60%">
</p>

<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> Folder Structure</h2>

    data
    .
    │
    ├── dataset50%
        ├── 64px
        │   │
        │   ├── train
        │   │   ├── benign
        │   │   ├── malignant
        │   │   
        │   ├── validation
        │   │   ├── benign
        │   │   ├── malignant
        │   │
        │   ├── test
        │       ├── benign
        │       ├── malignant
        │
        ├── 128px
        │   │
        │   ├── train
        │   │   ├── benign
        │   │   ├── malignant
        │   │   
        │   ├── validation
        │   │   ├── benign
        │   │   ├── malignant
        │   │
        │   ├── test
        │       ├── benign
        │       ├── malignant
        │
        ├── 256px
            │
            ├── train
            │   ├── benign
            │   ├── malignant
            │   
            ├── validation
            │   ├── benign
            │   ├── malignant
            │
            ├── test
                ├── benign
                ├── malignant

<p>  :house: <a href="#table-of-contents">  Back to Table of Contents</a> </p>


<!-- PREREQUISITES -->
<h2 id="prerequisites">Prerequisites</h2>

<p align="center">
  <a href="https://www.linux.org/">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="60px"/>
  </a>
  <a href="https://ubuntu.com/">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/ubuntu/ubuntu-plain-wordmark.svg" width="60px"/>
  </a>
  <a href="https://www.microsoft.com/en-ca/software-download/windows10">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/windows8/windows8-original.svg" width="60px"/>
  </a>
  <a href="https://www.tensorflow.org/">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="60px"/>
  </a>
  <a href="https://www.anaconda.com/">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/anaconda/anaconda-original.svg" width="60px"/>
  </a>
  <a href="https://www.python.org">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="70px"/>
  </a>
  <a href="https://opencv.org/" target="_blank">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original-wordmark.svg" width="70px"/>
  </a>
  <a href="https://numpy.org/" target="_blank">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="70px"/>
  </a>
  <a href="https://jupyter.org/" target="_blank">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original-wordmark.svg" width="70px"/>
  </a>
  <a href="https://pandas.pydata.org/" target="_blank">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="70px"/>
  </a>
<P/>

 1. Open a `Command Prompt` (NOT `Windows PowerShell`) or a `Terminal`
 2. Create a conda environment `conda create -n is python=3.6.6 -y`
 3. Activate this environment `activate is` (Windows) or `source activate is` (Linux/macOS)
 4. Install the following packages `tensorflow`, `keras`, `opencv`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, and `notebook`:       </br></br>
      * <a href="https://www.python.org/" target="_blank">Python (3.6)</a>
      * <a href="https://www.tensorflow.org/" target="_blank">TensorFlow (2.3.2)</a> - `pip install tensorflow==2.3.2`
      * <a href="https://keras.io/" target="_blank">Keras (2.1.6)</a> - `pip install keras==2.1.6`
      * <a href="https://opencv.org/" target="_blank">OpenCV (4.4.0)</a> - `pip install opencv==4.0.0`
      * <a href="https://matplotlib.org/" target="_blank">Matplotlib (3.5.1)</a> - `pip install matplotlib==3.5.1`
      * <a href="https://numpy.org/" target="_blank">NumPy (1.18.4)</a> - `pip install numpy==1.18.4`
      * <a href="https://pandas.pydata.org/" target="_blank">Pandas (1.1.3)</a> - `pip install pandas==1.1.3` 
      * <a href="https://scikit-learn.org/stable/" target="_blank">Sklearn (0.23.2)</a> - `pip install scikit-learn==0.23.2`
      * <a href="https://jupyter.org/" target="_blank">Jupyter (4.7.1)</a> - `pip install notebook`
</br>
<p>  :house: <a href="#table-of-contents">  Back to Table of Contents</a> </p>


<!-- DATASET -->
<h2 id="Dataset"> Dataset</h2>

<p align="justify">
The target data set investigated in this research are digital cytology images of breast cancer, which is an archival collection of samples taken from patients at the Regional Hospital in Zielona Gora, Poland. The data set consists of 550 microscopic images of cytological specimens taken from 50 patients using FNB without aspiration (under ultrasonography support) with 0.5-millimeter needles. To form a data set, cytological material extracted from the patient is digitized into virtual slides using the Olympus VS120 Virtual Microscopic System. A virtual slide is a massive digital image with an average size of 200,000 x 100,000 pixels. Since not all parts of a slide necessarily contain useful medical information for analysis, a cytologist manually selected 11 distinct regions of interest (ROI) which were converted to 8 bit/channel RGB TIFF files of size 1583 x 828 pixels.
</p> 

<p align="center"> 
  <img src="images/BreakHis.png" alt="BreakHis">
</p>

<p>  :house: <a href="#table-of-contents">  Back to Table of Contents</a> </p>

<!-- LICENSE -->
<h2 id="license"> License</h2>
© <a href="https://www.concordia.ca/research/cenparmi.html" target="_blank">CENPARMI</a> Lab - This code is made available under the <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">GPLv3</a> License and is available for non-commercial academic purposes.

<p>  :house: <a href="#table-of-contents">  Back to Table of Contents</a> </p>

<!-- REFERENCES -->
<h2 id="reference"> License</h2>
(BibTex ...)

<p>  :house: <a href="#table-of-contents">  Back to Table of Contents</a> </p>
