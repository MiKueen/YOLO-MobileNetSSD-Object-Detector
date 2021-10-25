
People Counter using YOLO and MobileNetSSD
==========================================

This project counts the no of people crossing a line and detects their direction of movement.
It can be used either with YOLO object detector or pretrained MobileNetSSD detector. 


Usage
~~~~~

**YOLO**

.. code-block:: bash

	cd yolo

	# First download the pretrained weights by running the command
	wget https://pjreddie.com/media/files/yolov3.weights

	# To read and write back out to video:
	python counter.py --input <path/to/input/video/file> --output <path/to/output/folder>/yolo.avi

	# To read from webcam and write back out to disk:
	python counter.py --output <path/to/output/folder>/yolo_webcam.avi


**MobileNetSSD**

.. code-block:: bash

	cd mobilenet_ssd

	# To read and write back out to video:
	python counter.py --input <path/to/input/video/file> --output <path/to/output/folder>/mobilenet_ssd.avi
	
	# To read from webcam and write back out to disk:
	python counter.py --output <path/to/output/folder>/mobilenet_ssd_webcam.avi

