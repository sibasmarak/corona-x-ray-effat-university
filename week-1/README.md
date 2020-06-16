## This folder contains two file:   
1. **segmentation.py**:
- This is used to segment the x-ray images obtained from various sources to three folders, namely, normal, corona and pneumonia.  
  - normal: x-ray with no finding
  - corona: x-ray with corona finding
  - pneumonia: x-ray with pneumonia finding   
2. **fine_tuning.py**:  
- The code uses InceptionV3 Model for fine tuning  
- Add:  
  - One Dense layer of 1024 units, relu activation, trainable
  - One Dense layer of 128 units, relu activation, trainable
  - Final Dense layer of 3 units (corresponding to normal, covid-19 and pneumonia), softmax activation, trainable  
- Fine tune:  
  - The topmost layer of InceptionV3 i.e. the layer just before the newly added Dense layer of 1024 units
