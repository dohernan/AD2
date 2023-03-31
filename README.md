
# SDCND : Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking. 

In this project, you'll fuse measurements from LiDAR and camera and track vehicles over time. You will be using real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

<img src="img/img_title_1.jpeg"/>

The project consists of two major parts: 
1. **Object detection**: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. 
2. **Object tracking** : In this part, an extended Kalman filter is used to track vehicles over time, based on the lidar detections fused with camera detections. Data association and track management are implemented as well.

The following diagram contains an outline of the data flow and of the individual steps that make up the algorithm. 

<img src="img/img_title_2_new.png"/>

Also, the project code contains various tasks, which are detailed step-by-step in the code. More information on the algorithm and on the tasks can be found in the Udacity classroom. 

## Project File Structure

📦project<br>
 ┣ 📂dataset --> contains the Waymo Open Dataset sequences <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> plot functions for tracking visualization and RMSE calculation<br>
 ┃ ┣ helpers.py --> misc. helper functions, e.g. for loading / saving binary files<br>
 ┃ ┗ objdet_tools.py --> object detection functions without student tasks<br>
 ┃ ┗ params.py --> parameter file for the tracking part<br>
 ┃ <br>
 ┣ 📂results --> binary files with pre-computed intermediate results<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> data association logic for assigning measurements to tracks incl. student tasks <br>
 ┃ ┣ filter.py --> extended Kalman filter implementation incl. student tasks <br>
 ┃ ┣ measurements.py --> sensor and measurement classes for camera and lidar incl. student tasks <br>
 ┃ ┣ objdet_detect.py --> model-based object detection incl. student tasks <br>
 ┃ ┣ objdet_eval.py --> performance assessment for object detection incl. student tasks <br>
 ┃ ┣ objdet_pcl.py --> point-cloud functions, e.g. for birds-eye view incl. student tasks <br>
 ┃ ┗ trackmanagement.py --> track and track management classes incl. student tasks  <br>
 ┃ <br>
 ┣ 📂tools --> external tools<br>
 ┃ ┣ 📂objdet_models --> models for object detection<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> functions for light-weight loading of Waymo sequences<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## Installation Instructions for Running Locally
### Cloning the Project
In order to create a local copy of the project, please click on "Code" and then "Download ZIP". Alternatively, you may of-course use GitHub Desktop or Git Bash for this purpose. 

### Python
The project has been written using Python 3.7. Please make sure that your local installation is equal or above this version. 

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. You may either install them one-by-one using pip or you can use the following command to install them all at once: 
`pip3 install -r requirements.txt` 

### Waymo Open Dataset Reader
The Waymo Open Dataset Reader is a very convenient toolbox that allows you to access sequences from the Waymo Open Dataset without the need of installing all of the heavy-weight dependencies that come along with the official toolbox. The installation instructions can be found in `tools/waymo_reader/README.md`. 

### Waymo Open Dataset Files
This project makes use of three different sequences to illustrate the concepts of object detection and tracking. These are: 
- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to register with Waymo Open Dataset first: [Open Dataset – Waymo](https://waymo.com/open/terms), if you have not already, making sure to note "Udacity" as your institution.

Once you have done so, please [click here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) to access the Google Cloud Container that holds all the sequences. Once you have been cleared for access by Waymo (which might take up to 48 hours), you can download the individual sequences. 

The sequences listed above can be found in the folder "training". Please download them and put the `tfrecord`-files into the `dataset` folder of this project.


### Pre-Trained Models
The object detection methods used in this project use pre-trained models which have been provided by the original authors. They can be downloaded [here](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) (darknet) and [here](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing) (fpn_resnet). Once downloaded, please copy the model files into the paths `/tools/objdet_models/darknet/pretrained` and `/tools/objdet_models/fpn_resnet/pretrained` respectively.

### Using Pre-Computed Results

In the main file `loop_over_dataset.py`, you can choose which steps of the algorithm should be executed. If you want to call a specific function, you simply need to add the corresponding string literal to one of the following lists: 

- `exec_data` : controls the execution of steps related to sensor data. 
  - `pcl_from_rangeimage` transforms the Waymo Open Data range image into a 3D point-cloud
  - `load_image` returns the image of the front camera

- `exec_detection` : controls which steps of model-based 3D object detection are performed
  - `bev_from_pcl` transforms the point-cloud into a fixed-size birds-eye view perspective
  - `detect_objects` executes the actual detection and returns a set of objects (only vehicles) 
  - `validate_object_labels` decides which ground-truth labels should be considered (e.g. based on difficulty or visibility)
  - `measure_detection_performance` contains methods to evaluate detection performance for a single frame

In case you do not include a specific step into the list, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be loaded using [this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing) link. Please use the folder `darknet` first. Unzip the file within and put its content into the folder `results`.

- `exec_tracking` : controls the execution of the object tracking algorithm

- `exec_visualization` : controls the visualization of results
  - `show_range_image` displays two LiDAR range image channels (range and intensity)
  - `show_labels_in_image` projects ground-truth boxes into the front camera image
  - `show_objects_and_labels_in_bev` projects detected objects and label boxes into the birds-eye view
  - `show_objects_in_bev_labels_in_camera` displays a stacked view with labels inside the camera image on top and the birds-eye view with detected objects on the bottom
  - `show_tracks` displays the tracking results
  - `show_detection_performance` displays the performance evaluation based on all detected 
  - `make_tracking_movie` renders an output movie of the object tracking results

Even without solving any of the tasks, the project code can be executed. 

The final project uses pre-computed lidar detections in order for all students to have the same input data. If you use the workspace, the data is prepared there already. Otherwise, [download the pre-computed lidar detections](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB), unzip them and put them in the folder `results`.

## External Dependencies
Parts of this project are based on the following repositories: 
- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License
[License](LICENSE.md)


Step-1: Compute Lidar point cloud from Range Imag


The initial step involves generating a Lidar point cloud from a range image. This is accomplished by initially examining the range image and converting its range and intensity channels into an 8-bit format. Subsequently, we utilize the openCV library to stack the range and intensity channels vertically, allowing us to visualize the image.

To accomplish this, we convert the "range" channel to 8-bit format and also convert the "intensity" channel to 8-bit format. We then proceed to crop the range image to include only the area within +/- 90 degrees to the left and right of the forward-facing x-axis. Finally, we use the openCV library to vertically stack the range and intensity channels, and these changes are implemented in the 'loop_over_dataset.py' script.

result: 

range image
![image](https://user-images.githubusercontent.com/38068231/228829523-99c435c0-5744-4e26-a101-b40661c184ee.png)
![image](https://user-images.githubusercontent.com/38068231/228829965-aaefbcbc-9f77-4cec-93e8-c38eb0141521.png)

point cloud
![image](https://user-images.githubusercontent.com/38068231/228829886-bb1d7ee8-617a-4746-aa7c-55508644ff20.png)

Examples of vehicles:
![image](https://user-images.githubusercontent.com/38068231/229065098-84386f6e-4fa8-4e5c-a8b0-e2a8ee87cb65.png)
![image](https://user-images.githubusercontent.com/38068231/229065177-aa0abc01-7fd3-42d6-8a41-5a531551f918.png)
![image](https://user-images.githubusercontent.com/38068231/229065245-368f39db-d678-4c2b-86fa-a31e6a5c0085.png)
![image](https://user-images.githubusercontent.com/38068231/229065269-b0d61b3b-1ca0-4e80-90cf-10137002b5c2.png)

By analyzing the distribution and arrangement of these point cloud views, we can identify the stable features of the vehicle that are common across different viewpoints.
The rear bumper is a stable feature that can be easily recognized since it is typically a distinctive shape and often located at the rear of the vehicle. The tires are also stable features since they are round and located at the corners of the vehicle. Side mirrors are also often identifiable due to their unique shape and location.
The chassis is the most frequent feature to be identified since it is the fundamental structure of the vehicle.


Step-2: Create Birds-Eye View from Lidar PCL

Firstly, the coordinates are transformed into their corresponding pixel values. Following this, the lidar intensity values are assigned to the bird's eye view (BEV) mapping. The sorted and pruned point cloud lidar obtained from the previous task is utilized here. The height map present in the BEV is normalized before proceeding to compute and map the intensity values. These steps are undertaken to generate a comprehensive BEV representation of the scene, which can be used for further analysis and processing.

point cloud:
![image](https://user-images.githubusercontent.com/38068231/228831012-4bcedc61-6d8d-48d0-a3b8-e7bcb7e0f20f.png)

Compute intensity layer of the BEV map:
![image](https://user-images.githubusercontent.com/38068231/228831543-398d9701-d43c-4487-ad4c-15055129d944.png)

Section 3 : Model-based Object Detection in BEV Image

Initially, we clone the repository and focus on the 'test.py' file for the current task. The necessary configurations are extracted from 'parse_test_configs()' and included in the 'load_configs_model' configuration structure. This is done to streamline the process of using the fpn resnet model from the cloned repository.

Once the model is instantiated, the 3D bounding boxes are extracted from the responses. These boxes are then converted from pixel coordinates to vehicle coordinates. The model output is configured to be in the bounding box format [class-id, x, y, z, h, w, l, yaw] for ease of use and further processing.
result:
![image](https://user-images.githubusercontent.com/38068231/228832406-3d4bcaf0-1a67-49d3-a7a7-ce7ae6cebc69.png)

Section 4 : Performance Evaluation for Object Detection

The current step involves computing the performance of the detection model by determining the intersection over union (IOU) between the labels and detections. False positive and false negative values are then computed based on these results. The primary task is to establish the geometric overlap between the bounding boxes of the labels and detected objects.

To accomplish this, detected objects are assigned to labels if the IOU exceeds a specified threshold. The degree of geometric overlap between the bounding boxes is computed to identify the best matches for multiple objects/detections. These matches are selected based on the maximum IOU. Finally, the false negative and false positive values are calculated. Precision and recall are also determined based on the false positive and false negative values, providing a comprehensive understanding of the model's performance.

results:
![image](https://user-images.githubusercontent.com/38068231/228836094-45cc2e41-399d-4650-8d1d-504e86874f53.png)

results with configs_det.use_labels_as_objects=True
![image](https://user-images.githubusercontent.com/38068231/228836712-00ee6b33-06fb-4b39-96cc-7ad09b7174a1.png)

