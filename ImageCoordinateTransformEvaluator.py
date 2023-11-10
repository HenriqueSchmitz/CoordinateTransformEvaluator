from TensorflowImageDetector import TensorflowImageDetector
from sys import platform
import numpy as np
import cv2
import json
import math

class ImageCoordinateTransformEvaluator:
  def __init__(self, pixelsToDistanceFactor, frameName = "Evaluated Frame", mapName = "Detections Map", frameScale=1, desiredDetectionLabel = "robot"):
    self.__frameName = frameName
    self.__mapName = mapName
    self.__frameScale = frameScale
    self.__isEvaluationDone = False
    self.__pixelsToDistanceFactor = pixelsToDistanceFactor
    self.__upArrowCode = 2490368 if platform == "win32" else 63232
    self.__downArrowCode = 2621440 if platform == "win32" else 63233
    self.__leftArrowCode = 2424832 if platform == "win32" else 63234
    self.__rightArrowCode = 2555904 if platform == "win32" else 63235
    self.__backspaceCode = 8 if platform == "win32" else 127
    self.__enterCode = 13
    self.__mapImage = None
    self.__userPoints = []
    self.__detectedPoints = []
    self.__transform = None
    self.__isDisplayingDetectedPoints = False
    self.__desiredDetectionLabel = desiredDetectionLabel

  def evaluateImage(self, mapImage, zipArchive, imagePath, transform, blockBoxes):
    self.__isEvaluationDone = False
    self.__mapImage = mapImage
    self.__userPoints = []
    self.__detectedPoints = []
    self.__transform = transform
    self.__isDisplayingDetectedPoints = False
    self.__zipArchive = zipArchive
    self.__imagePath = imagePath
    self.__blockBoxes = blockBoxes
    self.__frame = self.__readImageFromZip(zipArchive,imagePath)
    self.__detector = TensorflowImageDetector()
    self.__detections = self.__detector.loadDetectionsFromZip(zipArchive, imagePath.replace(".jpg", ".json"))
    self.__relevantDetections = [detection for detection in self.__detections if detection["label"] == self.__desiredDetectionLabel]
    self.__relevantDetections = self.__detector.filterOutDetectionsInBlockedBoxes(self.__relevantDetections, blockBoxes)
    self.__ingoredDetectionIndices = []
    self.__updateFrame()
    self.__updateMap()
    cv2.setMouseCallback(self.__mapName, self.__processMouseClick)
    cv2.setMouseCallback(self.__frameName, self.__processFrameMouseClick)
    while not self.__isEvaluationDone:
      key = cv2.waitKeyEx(1)
      self.__processKeyStroke(key)

  def __rescaleImage(self, image, scalePercentage):
    width = int(image.shape[1] * scalePercentage)
    height = int(image.shape[0] * scalePercentage)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

  def __readImageFromZip(self, zip, imagePath):
      imageData = zip.read(imagePath)
      image = cv2.imdecode(np.frombuffer(imageData, np.uint8), 1)
      return image
  
  def __updateMap(self):
    editedMap = self.__mapImage.copy()
    for point in self.__userPoints:
      cv2.circle(editedMap, tuple(point), 5, (0, 255, 0), -1)
    for point in self.__detectedPoints:
      cv2.circle(editedMap, tuple(point["mapPoint"]), 5, (255, 255, 255), -1)
    cv2.imshow(self.__mapName, editedMap)

  def __updateFrame(self):
    nonIgnoredDetections = self.__calculateNonIgnoredDetections()
    imageWithDetections = self.__detector.drawDetectionsOnImage(self.__frame, nonIgnoredDetections, ["robot"], self.__blockBoxes, drawMidpoint = True, yCenterFromTopLeft=0.67)
    rescaledTestedImage = self.__rescaleImage(imageWithDetections, self.__frameScale)
    cv2.imshow(self.__frameName, rescaledTestedImage)
  
  def __calculateNonIgnoredDetections(self):
    nonIgnoredDetections = []
    for index, detection in enumerate(self.__relevantDetections):
      if index not in self.__ingoredDetectionIndices:
        nonIgnoredDetections.append(detection)
    return nonIgnoredDetections\

  def __processMouseClick(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
      if not self.__isDisplayingDetectedPoints:
        self.__userPoints.append([x,y])
        self.__updateMap()

  def __processFrameMouseClick(self, event, x, y, flags, param):
    x = round(x / self.__frameScale)
    y = round(y / self.__frameScale)
    if event == cv2.EVENT_LBUTTONUP:
      if not self.__isDisplayingDetectedPoints:
        minimumDistance = None
        closestIndex = None
        clickedPoint = [x,y]
        for index, detection in enumerate(self.__calculateNonIgnoredDetections()):
          robotBase = self.__detector.getMidpoint(detection, yCenterFromTopLeft=0.67)
          distance = self.__calculateDistanceBetweenPoints(clickedPoint, robotBase)
          if minimumDistance == None or distance < minimumDistance:
            minimumDistance = distance
            if distance < 15:
              closestIndex = index
        self.__ingoredDetectionIndices.append(closestIndex)
        self.__updateFrame()

  def __processKeyStroke(self, key):
    if key == ord("s"):
      print("Skipping image based on user input")
      self.__isEvaluationDone = True
    if key == ord("b"):
      if not self.__isDisplayingDetectedPoints:
        if len(self.__ingoredDetectionIndices) > 0:
          self.__ingoredDetectionIndices.pop()
          self.__updateFrame()
    if key == self.__backspaceCode:
      if not self.__isDisplayingDetectedPoints:
        if len(self.__userPoints) > 0:
          self.__userPoints.pop()
          self.__updateMap()
      else:
        self.__isDisplayingDetectedPoints = False
        self.__detectedPoints = []
        self.__updateMap()
    elif key == self.__enterCode:
      if not self.__isDisplayingDetectedPoints:
        self.__isDisplayingDetectedPoints = True
        for detection in self.__calculateNonIgnoredDetections():
          robotBase = self.__detector.getMidpoint(detection, yCenterFromTopLeft=0.67)
          transformedDetection = self.__transform.roundedShiftPerspectiveForPoint(robotBase)
          detectedPoint = {}
          detectedPoint["mapPoint"] = transformedDetection
          detectedPoint["usedRobotBase"] = robotBase
          detectedPoint["topLeftPoint"] = detection["topLeftPoint"]
          detectedPoint["bottomRightPoint"] = detection["bottomRightPoint"]
          self.__detectedPoints.append(detectedPoint)
        self.__updateMap()
      else:
        if len(self.__userPoints) != len(self.__detectedPoints):
          print("Number of detected and provided points do not match!!")
        else:
          resultFilePath = self.__imagePath.replace(".jpg", "_evaluation_points.json")
          evaluationResults = {}
          evaluationResults["userProvidedPoints"] = self.__userPoints
          evaluationResults["transformResultPoints"] = [ detectedPoint["mapPoint"] for detectedPoint in self.__detectedPoints]
          evaluationResults["pointPairs"] = self.__calculatePointPairs()
          evaluationResultsJson = json.dumps(evaluationResults)
          with self.__zipArchive.open(resultFilePath, 'w') as file:
            file.write(evaluationResultsJson.encode())
            print("Saved evaluation points: " + resultFilePath)
          self.__isEvaluationDone = True

  def __calculateDistanceBetweenPoints(self, pointA, pointB):
    return math.sqrt(math.pow(pointA[0] - pointB[0], 2) + math.pow(pointA[1] - pointB[1], 2))
  
  def __calculatePointPairs(self):
    pairs = []
    for detectedPoint in self.__detectedPoints:
      pairedPoint = [0,0]
      minimumDistance = None
      for userPoint in self.__userPoints:
        distance = self.__calculateDistanceBetweenPoints(detectedPoint["mapPoint"], userPoint)
        if minimumDistance is None or distance < minimumDistance:
          pairedPoint = userPoint
          minimumDistance = distance
      pixelDistance = minimumDistance
      measurementDistance = pixelDistance * self.__pixelsToDistanceFactor
      pair = {}
      pair["detectedPoint"] = detectedPoint
      pair["userPoint"] = pairedPoint
      pair["pixelDistance"] = pixelDistance
      pair["measurementDistance"] = measurementDistance
      pairs.append(pair)
    return pairs