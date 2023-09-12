from TensorflowImageDetector import TensorflowImageDetector
from sys import platform
import numpy as np
import cv2

class ImageCoordinateTransformEvaluator:
  def __init__(self, frameName = "Evaluated Frame", mapName = "Detections Map", frameScale=1, desiredDetectionLabel = "robot"):
    self.__frameName = frameName
    self.__mapName = mapName
    self.__frameScale = frameScale
    self.__isEvaluationDone = False
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
    testedImage = self.__readImageFromZip(zipArchive,imagePath)
    self.__detector = TensorflowImageDetector()
    self.__detections = self.__detector.loadDetectionsFromZip(zipArchive, imagePath.replace(".jpg", ".json"))
    imageWithDetections = self.__detector.drawDetectionsOnImage(testedImage, self.__detections, ["robot"], blockBoxes, drawMidpoint = True, yCenterFromTopLeft=0.67)
    rescaledTestedImage = self.__rescaleImage(imageWithDetections, self.__frameScale)
    cv2.imshow(self.__frameName, rescaledTestedImage)
    self.__updateMap()
    cv2.setMouseCallback(self.__mapName, self.__processMouseClick)
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
      cv2.circle(editedMap, tuple(point), 5, (255, 255, 255), -1)
    cv2.imshow(self.__mapName, editedMap)

  
  def __processMouseClick(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
      if not self.__isDisplayingDetectedPoints:
        self.__userPoints.append([x,y])
        self.__updateMap()

  def __processKeyStroke(self, key):
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
        for detection in self.__detections:
          if detection["label"] == self.__desiredDetectionLabel:
            robotBase = self.__detector.getMidpoint(detection, yCenterFromTopLeft=0.67)
            transformedDetection = self.__transform.roundedShiftPerspectiveForPoint(robotBase)
            self.__detectedPoints.append(transformedDetection)
        self.__updateMap()
      else:
        if len(self.__userPoints) != len(self.__detectedPoints):
          print("Number of detected and provided points do not match!!")
        else:
          self.__isEvaluationDone = True