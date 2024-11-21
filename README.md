The AI-Driven Face Liveness Detector enhances the security of face recognition systems by preventing face spoofing through a web-based solution that detects real, live faces. 
Using TensorFlow, Keras, OpenCV, and MediaPipe, the model distinguishes between real and fake faces, minimizing false positives and ensuring reliable authentication. 
Integrated with React and TensorFlow JS for seamless deployment, it employs MongoDB and FastAPI for database and API management. 
The solution is scalable, accessible across devices, and provides a cost-effective, secure alternative to traditional face recognition methods.


## How to run the  project?

Steps to run :
1. Use DataCollector to start the camera and take images of automatically detected images.
2. Save images will be stord in DATA/Dataset
3. Take images of Fake/Real faces first and move them form DATA/Dataset to DATA/Preprocess/fake or DATA/Preprocess/real respectively.
4. Then collect images for reamaining section and move them into respective folder in DATA/Preprocessed
5. The run the  'architecture.py' file.
6. After that run 'detector2.py' to open camera and start face livensss detection.


## Demo

Demo: https://youtu.be/UAdaqm4e5yo

