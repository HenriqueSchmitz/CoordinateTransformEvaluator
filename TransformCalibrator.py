import cv2
from typing import List
import numpy as np
from DistortionAdjustedCoordinateTransform import DistortionAdjustedCoordinateTransform
from sys import platform

class TransformCalibrator:
  def __init__(self, mapReference, background, imageName = "Calibration"):
    self.__mapReferenceTemplate = mapReference
    self.__backgroundTemplate = background
    self.__mapReference = mapReference
    self.__background = background
    self.__imageName = imageName
    self.__mapPoints = []
    self.__backgroundPoints = []
    self.__pointNumbersForDistortionCorrection = []
    self.__pointNumbersForTransform = []
    self.__allPointsAdded = False
    self.__calibrationCompleted = False
    self.__currentPoint = 0
    self.__currentLabel = None
    self.__upArrowCode = 2490368 if platform == "win32" else 63232
    self.__downArrowCode = 2621440 if platform == "win32" else 63233
    self.__leftArrowCode = 2424832 if platform == "win32" else 63234
    self.__rightArrowCode = 2555904 if platform == "win32" else 63235
    self.__backspaceCode = 8 if platform == "win32" else 127
    self.__enterCode = 13
    self.__transform = None
    self.__testPoint = None
    self.__convertedTestPoint = None

  def run(self, mapPoints, pointNumbersForTransform, pointNumbersForDistortionCorrection):
    self.__mapPoints = mapPoints
    self.__currentLabel = self.__mapPoints[self.__currentPoint]["label"]
    self.__pointNumbersForDistortionCorrection = pointNumbersForDistortionCorrection
    self.__pointNumbersForTransform = pointNumbersForTransform
    self.__updateView()
    cv2.setMouseCallback(self.__imageName, self.__addPoint)
    while not self.__calibrationCompleted:
      key = cv2.waitKeyEx(1)
      self.__processArrowKeys(key)
      self.__processCommandKeys(key)
    cv2.destroyWindow(self.__imageName)
    return self.__transform

  def __updateView(self):
    self.__background = self.__backgroundTemplate.copy()
    self.__drawPointsOnImage(self.__background, self.__backgroundPoints)
    self.__mapReference = self.__mapReferenceTemplate.copy()
    self.__drawPointsOnImage(self.__mapReference, self.__mapPoints[:self.__currentPoint + 1])
    self.__drawTestPointOnImages()
    fullImage = self.__differentSizedVerticalConcat([self.__background, self.__mapReference])
    cv2.imshow(self.__imageName, fullImage)

  def __drawPointsOnImage(self, image, points):
    for point in points:
      cv2.circle(image, (point["x"], point["y"]), 3, (0, 255, 0), -1)
      if "label" in point and point["label"] != None:
        cv2.putText(image, point["label"], (point["x"], point["y"]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

  def __drawTestPointOnImages(self):
    if self.__testPoint is not None and self.__convertedTestPoint is not None:
      cv2.circle(self.__background, tuple(self.__testPoint), 3, (255, 255, 255), -1)
      cv2.circle(self.__background, tuple(self.__testPoint), 10, (255, 255, 255), 1)
      cv2.circle(self.__mapReference, tuple(self.__convertedTestPoint), 3, (255, 255, 255), -1)
      cv2.circle(self.__mapReference, tuple(self.__convertedTestPoint), 10, (255, 255, 255), 1)

  def __differentSizedVerticalConcat(self, images: List[List[List[List[int]]]]) -> List[List[List[int]]]:
    imageWidths = [image.shape[1] for image in images]
    maxWidth = max(imageWidths)
    fullWidthImages = []
    for image in images:
      imageWidth = image.shape[1]
      if imageWidth < maxWidth:
        missingWidth = maxWidth - imageWidth
        imageHeight = image.shape[0]
        fillerBlock = np.zeros((imageHeight, missingWidth, 3), dtype=np.uint8)
        resizedImage = cv2.hconcat([image, fillerBlock])
        fullWidthImages.append(resizedImage)
      else:
        fullWidthImages.append(image)
    return cv2.vconcat(fullWidthImages)
  
  def __addPoint(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
      if not self.__allPointsAdded:
        self.__backgroundPoints.append({"x": x, "y": y, "label": self.__currentLabel})
        self.__currentPoint = self.__currentPoint + 1
        if self.__currentPoint < len(self.__mapPoints):
          self.__currentLabel = self.__mapPoints[self.__currentPoint]["label"]
        else:
          self.__allPointsAdded = True
      if self.__transform is not None:
        self.__testPoint = [x, y]
        self.__convertedTestPoint = self.__transform.roundedShiftPerspectiveForPoint(self.__testPoint)
      self.__updateView()

  def __processArrowKeys(self, key):
    if key == self.__upArrowCode:
      self.__backgroundPoints[self.__currentPoint - 1]["y"] = self.__backgroundPoints[self.__currentPoint - 1]["y"] - 1
      self.__updateView()
    elif key == self.__downArrowCode:
      self.__backgroundPoints[self.__currentPoint - 1]["y"] = self.__backgroundPoints[self.__currentPoint - 1]["y"] + 1
      self.__updateView()
    elif key == self.__leftArrowCode:
      self.__backgroundPoints[self.__currentPoint - 1]["x"] = self.__backgroundPoints[self.__currentPoint - 1]["x"] - 1
      self.__updateView()
    elif key == self.__rightArrowCode:
      self.__backgroundPoints[self.__currentPoint - 1]["x"] = self.__backgroundPoints[self.__currentPoint - 1]["x"] + 1
      self.__updateView()

  def __processCommandKeys(self, key):
    if key == self.__backspaceCode:
      if self.__currentPoint > 0:
        self.__currentPoint = self.__currentPoint - 1
        self.__allPointsAdded = False
        self.__currentLabel = self.__mapPoints[self.__currentPoint]["label"]
        self.__backgroundPoints.pop()
        self.__transform = None
        self.__testPoint = None
        self.__convertedTestPoint = None
        self.__updateView()
    elif key == self.__enterCode:
      if self.__allPointsAdded:
        if self.__transform is None:
          distortionCorrectionPoints = []
          imageTransformPoints = []
          mapTransformPoints = []
          for point in self.__backgroundPoints:
            if point["label"] in self.__pointNumbersForDistortionCorrection:
              distortionCorrectionPoints.append([point["x"], point["y"]])
            if point["label"] in self.__pointNumbersForTransform:
              imageTransformPoints.append([point["x"], point["y"]])
          for point in self.__mapPoints:
            if point["label"] in self.__pointNumbersForTransform:
              mapTransformPoints.append([point["x"], point["y"]])
          backgroundCenter = [self.__backgroundTemplate.shape[1]/2, self.__backgroundTemplate.shape[0]/2]
          self.__transform = DistortionAdjustedCoordinateTransform(distortionCorrectionPoints, backgroundCenter, imageTransformPoints, mapTransformPoints)
        else:
          self.__calibrationCompleted = True