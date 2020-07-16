#!/usr/bin/env python
# coding: utf-8

# Einkommentieren, falls nur CPU genutzt werden soll

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import urllib
from datetime import datetime
import boto3
import json
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

# Geklaut von https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
        
        graph = tf.compat.v1.get_default_graph()
        for x in self.default_graph.get_operations():
            x.name        

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('model_outputs:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()



class PeopleCounter:
    def __init__(self, model_path, threshold=0.7):
        self.odapi = DetectorAPI(path_to_ckpt=model_path)
        self.threshold = threshold

    def get_image(self, url, id):
        resp = urllib.request.urlopen(url)
        self.image = np.asarray(bytearray(resp.read()), dtype="uint8")
        #if self.img is not None:
        self.image = cv2.imdecode(self.image, -1)
        #status = cv2.imwrite("/tmp/"+ str(id) + ".jpg", self.image)
        #print("Image written to file-system : ",status)       

    def count_people(self, verbose=False):
        peoplecount = 0
        boxes, scores, classes, num = self.odapi.processFrame(self.image)
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                cv2.rectangle(self.image,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                peoplecount += 1
            if verbose:
                cv2.imshow('image', self.image)
                cv2.waitKey(0)
        return peoplecount


if __name__ == '__main__':
    model_path = './azure_general_webcam/model.pb'
    with open("webcam_list.json","r") as f:
        webcams = json.load(f)
    pc = PeopleCounter(model_path)

    for cam in webcams:
        try:
            pc.get_image(cam['URL'], cam['ID'])
            cam['Personenzahl'] = pc.count_people(verbose=False)
            cam['Stand'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(cam["Name"]+" :"+str(cam["Personenzahl"]))        
        except urllib.error.HTTPError as e:
            print(cam["Name"]+" :"+'The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except urllib.error.URLError as e:
            print(cam["Name"]+" :"+'We failed to reach a server.')
            print('Reason: ', e.reason)
        except:
            pass

    client_s3 = boto3.client("s3" )

    response = client_s3.put_object(
        Bucket="sdd-s3-bucket",
        Body=json.dumps(webcams),
        Key=f"webcamdaten/{datetime.now().strftime('%Y/%m/%d/%H')}webcamdaten2.json"
      )
    
    #directory = r'/tmp'
    #for filename in os.listdir(directory):
    #    if filename.endswith(".jpg"):
    #        print(os.path.join(directory, filename))
    #        s3 = boto3.resource('s3')
    #        s3.Bucket('sdd-s3-bucket').upload_file(os.path.join(directory, filename), f"webcampictures/{datetime.now().strftime('%Y/%m/%d/%H')}" + "/" + filename)  
    #    else:
    #        continue
    
