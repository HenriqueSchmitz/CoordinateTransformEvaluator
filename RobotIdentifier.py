import numpy as np
import cv2

class RobotIdentifier:
  def __init__(self):
    self.__blueLower = np.array([100, 40, 40], np.uint8)
    self.__blueUpper = np.array([140, 180, 180], np.uint8)
    self.__red1Lower = np.array([0, 20, 20], np.uint8)
    self.__red1Upper = np.array([15, 200, 200], np.uint8)
    self.__red2Lower = np.array([165, 20, 20], np.uint8)
    self.__red2Upper = np.array([180, 200, 200], np.uint8)

  def identifyRobot(self, evaluatedImage, detection):
    croppedBumper = self.__cropRobotBumper(evaluatedImage, detection)
    hsvBumper = cv2.cvtColor(croppedBumper, cv2.COLOR_BGR2HSV)
    blueValue = self.__calculateBlueValue(hsvBumper)
    redValue = self.__calculateRedValue(hsvBumper)
    if(blueValue > redValue and blueValue > 50):
      return "blue"
    elif(redValue > blueValue and redValue > 50):
      return "red"
    else:
      return "unknown"

  def __calculateBlueValue(self, hsvBumper):
    blueMask = cv2.inRange(hsvBumper, self.__blueLower, self.__blueUpper)
    blueValue = np.average(blueMask)
    return blueValue

  def __calculateRedValue(self, hsvBumper):
    red1Mask = cv2.inRange(hsvBumper, self.__red1Lower, self.__red1Upper)
    red2Mask = cv2.inRange(hsvBumper, self.__red2Lower, self.__red2Upper)
    redMask = cv2.bitwise_or(red1Mask, red2Mask)
    redValue = np.average(redMask)
    return redValue

  def __cropRobotBumper(self, image, detection, bumperHeightPercentage = 0.33):
    height = detection["bottomRightPoint"][1] - detection["topLeftPoint"][1]
    yMin = round(detection["bottomRightPoint"][1] - height * bumperHeightPercentage)
    topLeftPoint = [detection["topLeftPoint"][0], yMin]
    return self.__crop(image, topLeftPoint, detection["bottomRightPoint"])

  def __crop(self, image, topLeftPoint, bottomRightPoint):
    xMin = topLeftPoint[0]
    yMin = topLeftPoint[1]
    xMax = bottomRightPoint[0]
    yMax = bottomRightPoint[1]
    return image[yMin:yMax, xMin:xMax].copy()