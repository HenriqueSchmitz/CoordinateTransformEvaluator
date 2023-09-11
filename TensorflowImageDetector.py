import cv2
# import tensorflow as tf
# import numpy as np
# from typing import List
import json

class TensorflowImageDetector:
  def __init__(self):
    self.__isModelLoaded = False

  # def loadDetectionModel(self, pathToModel, pathToLabelMap):
  #   self.__detectionModel = tf.saved_model.load(pathToModel)
  #   # self.__categoryIndex = label_map_util.create_category_index_from_labelmap(pathToLabelMap, use_display_name=True)
  #   self.__categoryIndex = {}
  #   self.__isModelLoaded = True

  # def getDetectionBoundingBoxes(self, image: List[List[List[int]]], threshold: int, maxDetections: int):
  #   detections = self.__detectFromImage(image, self.__detectionModel)
  #   scores = detections['detection_scores'][0, :maxDetections].numpy()
  #   bboxes = detections['detection_boxes'][0, :maxDetections].numpy()
  #   labels = detections['detection_classes'][0, :maxDetections].numpy().astype(np.int64)
  #   labels = [self.__categoryIndex[n]['name'] for n in labels]
  #   (h, w, d) = image.shape
  #   detectedObjects = []
  #   for bbox, label, score in zip(bboxes, labels, scores):
  #     if score > threshold:
  #       xMin, yMin = int(bbox[1]*w), int(bbox[0]*h)
  #       xMax, yMax = int(bbox[3]*w), int(bbox[2]*h)
  #       topLeftPoint = [xMin, yMin]
  #       bottomRightPoint = [xMax, yMax]
  #       detectedObject = {
  #           "label": label,
  #           "topLeftPoint": topLeftPoint,
  #           "bottomRightPoint": bottomRightPoint,
  #           "score": score
  #       }
  #       detectedObjects.append(detectedObject)
  #   return detectedObjects

  # def __detectFromImage(self, image, model):
  #   if not self.__isModelLoaded:
  #     raise Exception("No model loaded")
  #   image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #   input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
  #   return model(input_tensor)

  def drawDetectionsOnImage(self, image, detections, allowedLabels = None, blockBoxes = [], xCenterFromTopLeft = 0.5, yCenterFromTopLeft = 0.5, drawMidpoint = False):
    editedImage = image.copy()
    if isinstance(allowedLabels, list):
      for detection in detections:
        if detection['label'] in allowedLabels:
          midpoint = self.__getMidpoint(detection, xCenterFromTopLeft, yCenterFromTopLeft)
          isBlocked = False
          for blockBox in blockBoxes:
            if self.__isPointInBox(midpoint, blockBox):
              isBlocked = True
          if not isBlocked:
            self.__drawSingleDetection(editedImage, detection)
          if drawMidpoint:
            cv2.circle(editedImage, midpoint, 2, (0,255,0), -1)
    else:
      for detection in detections:
        midpoint = self.__getMidpoint(detection, xCenterFromTopLeft, yCenterFromTopLeft)
        isBlocked = False
        for blockBox in blockBoxes:
          if self.__isPointInBox(midpoint, blockBox):
            isBlocked = True
        if not isBlocked:
          self.__drawSingleDetection(editedImage, detection)
        if drawMidpoint:
          cv2.circle(editedImage, midpoint, 2, (0,255,0), -1)
    return editedImage

  def __drawSingleDetection(self, image, detection):
    cv2.rectangle(image, detection["topLeftPoint"], detection["bottomRightPoint"], (0,255,0), 2)
    cv2.putText(image, f"{detection['label']}: {int(detection['score']*100)} %", detection["topLeftPoint"], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

  def __getMidpoint(self, detection, xCenterFromTopLeft = 0.5, yCenterFromTopLeft = 0.5):
    width = detection["bottomRightPoint"][0] - detection["topLeftPoint"][0]
    midX = round(detection["topLeftPoint"][0] + width * xCenterFromTopLeft)
    height = detection["bottomRightPoint"][1] - detection["topLeftPoint"][1]
    midY = round(detection["topLeftPoint"][1] + height * yCenterFromTopLeft)
    return (midX, midY)
  
  def __isPointInBox(self, point, box):
    minX = box["topLeftPoint"][0]
    minY = box["topLeftPoint"][1]
    maxX = box["bottomRightPoint"][0]
    maxY = box["bottomRightPoint"][1]
    if point[0] > minX and point[0] < maxX and point[1] > minY and point[1] < maxY:
      return True
    return False

  # def saveDetections(self, detections, filePath):
  #   convertedDetections = [{**detection, 'score': float(detection["score"])} for detection in detections]
  #   detectionsJson = json.dumps(convertedDetections)
  #   with open(filePath, 'w') as file:
  #     file.write(detectionsJson)
  
  def loadDetections(self, filePath):
    readFile = ""
    with open(filePath, 'r') as file:
      readFile = file.read()
    loadedDetections = json.loads(readFile)
    return loadedDetections
  
  def loadDetectionsFromZip(self, zipArchive, filePath):
    readFile = ""
    with zipArchive.open(filePath, 'r') as file:
      readFile = file.read()
    loadedDetections = json.loads(readFile)
    return loadedDetections