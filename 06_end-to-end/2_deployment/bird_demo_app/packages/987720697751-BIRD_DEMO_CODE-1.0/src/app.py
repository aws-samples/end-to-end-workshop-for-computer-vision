
import json
import logging
import time
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import panoramasdk
import datetime

class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.MODEL_NODE = "model_node"
        self.MODEL_DIM = 224
        self.frame_num = 0
        self.tracked_objects = []
        self.tracked_objects_start_time = dict()
        self.tracked_objects_duration = dict()
    
        self.classes = {
            0: "Bobolink", 
            1: "Cardinal", 
            2: "Purple_Finch",
            3: "Northern_Flicker",
            4:"American_Goldfinch",
            5:"Ruby_throated_Hummingbird",
            6:"Blue_Jay",
            7:"Mallard"   
        }

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        self.frame_num += 1
        logger.debug(self.frame_num)

        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream in streams:
            self.process_media(stream)

        self.outputs.video_out.put(streams)

    def process_media(self, stream):
        """Runs inference on a frame of video."""
        image_data = preprocess(stream.image, self.MODEL_DIM)
        logger.debug(image_data.shape)

        # Run inference
        inference_results = self.call({"input_1":image_data}, self.MODEL_NODE)

        # Process results (object deteciton)
        self.process_results(inference_results, stream)

    def process_results(self, inference_results, stream):
        """Processes output tensors from a computer vision model and annotates a video frame."""
        if inference_results is None:
            logger.warning("Inference results are None.")
            return
        
        logger.debug('Inference results: {}'.format(inference_results))
        count = 0
        for det in inference_results:
            if count == 0:
                first_output = det
            count += 1
            
        # first_output = inference_results[0]
        logger.debug('Output one type: {}'.format(type(first_output)))
        probabilities = first_output[0]
        # 1000 values for 1000 classes
        logger.debug('Result one shape: {}'.format(probabilities.shape))
        top_result = probabilities.argmax()
        
        self.detected_class = self.classes[top_result]
        self.detected_frame = self.frame_num
        # persist for up to 5 seconds
        # if self.frame_num - self.detected_frame < 75:
        label = '{} ({}%)'.format(self.detected_class, int(probabilities[top_result]*100))
        stream.add_label(label, 0.1, 0.1)

def preprocess(img, size):
    """Resizes and normalizes a frame of video."""
    resized = cv2.resize(img, (size, size))
    x1 = np.asarray(resized)
    x1 = np.expand_dims(x1, 0)
    return x1

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    try:
        logger.info("INITIALIZING APPLICATION")
        app = Application()
        logger.info("PROCESSING STREAMS")
        while True:
            app.process_streams()
    except Exception as e:
        logger.warning(e)

logger = get_logger(level=logging.INFO)
main()
