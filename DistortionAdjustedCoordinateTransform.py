from typing import List
import math
import torch

class DistortionAdjustedCoordinateTransform:
  def __init__(self, distortionLinePoints: List[List[float]], distortionCenter: List[float], seenReferencePoints: List[List[float]], targetReferencePoints: List[List[float]]):
    self.__distortionLinePoints = distortionLinePoints
    self.__distortionCenter = distortionCenter
    self.__seenReferencePoints = seenReferencePoints
    self.__targetReferencePoints = targetReferencePoints
    self.__findLensCorrectionCoefficient(distortionLinePoints, maxIterations = 100)
    self.__findUndistortedPerspectiveShift(seenReferencePoints, targetReferencePoints)

  def shiftPerspectiveForPoint(self, point: List[float]) -> List[float]:
    undistortedPoint = self.correctDistortionForPoint(point)
    return self.__transformPointPerspectiveOnly(undistortedPoint)
  
  def roundedShiftPerspectiveForPoint(self, point: List[float]) -> List[int]:
    shiftedFloatPoint = self.shiftPerspectiveForPoint(point)
    return [round(coordinate) for coordinate in shiftedFloatPoint]

  def correctDistortionForPoint(self, distortedPoint: List[float]) -> List[float]:
    rSq = self.__rSquared(distortedPoint)
    xd = distortedPoint[0]
    yd = distortedPoint[1]
    xc = self.__distortionCenter[0]
    yc = self.__distortionCenter[1]
    xu = xc + (xd - xc)/(1 + self.__distortionCorrectionFactor*rSq)
    yu = yc + (yd - yc)/(1 + self.__distortionCorrectionFactor*rSq)
    return [xu, yu]

  def __transformPointPerspectiveOnly(self, point: List[float]) -> List[int]:
    a = self.__perspectiveShiftCoefficients[0]
    b = self.__perspectiveShiftCoefficients[1]
    c = self.__perspectiveShiftCoefficients[2]
    d = self.__perspectiveShiftCoefficients[3]
    e = self.__perspectiveShiftCoefficients[4]
    f = self.__perspectiveShiftCoefficients[5]
    g = self.__perspectiveShiftCoefficients[6]
    h = self.__perspectiveShiftCoefficients[7]
    x = point[0]
    y = point[1]
    newX = (a*x + b*y + c) / (g*x + h*y + 1)
    newY = (d*x + e*y + f) / (g*x + h*y + 1)
    return [newX, newY]

  def __rSquared(self, point: List[float]) -> float:
    x = point[0]
    y = point[1]
    cx = self.__distortionCenter[0]
    cy = self.__distortionCenter[1]
    return math.pow(x - cx, 2) + math.pow(y - cy, 2)

  def __findLensCorrectionCoefficient(self, distortionLinePoints: List[List[float]], maxIterations: int = 100) -> None:
    left, middle, right = self.__orderPointsByY(distortionLinePoints)
    self.__distortionCorrectionFactor = -0.000001
    self.__distortionCorrectionFloor = None
    self.__distortionCorrectionCeiling = None
    for iteration in range(1, maxIterations):
      try:
        middleYError = self.__findOffsetToLineForUndistortedPoints(left, middle, right)
        if(abs(middleYError) <= 1):
          return
        elif(middleYError < 0):
          self.__distortionCorrectionCeiling = self.__distortionCorrectionFactor
          if(self.__distortionCorrectionFloor):
            difference = abs(self.__distortionCorrectionCeiling) - abs(self.__distortionCorrectionFloor)
            self.__distortionCorrectionFactor = self.__distortionCorrectionFloor - difference / 2
          else:
            self.__distortionCorrectionFactor = self.__distortionCorrectionCeiling / 2
        else:
          self.__distortionCorrectionFloor = self.__distortionCorrectionFactor
          if(self.__distortionCorrectionCeiling):
            difference = abs(self.__distortionCorrectionCeiling) - abs(self.__distortionCorrectionFloor)
            self.__distortionCorrectionFactor = self.__distortionCorrectionFloor - difference / 2
          else:
            self.__distortionCorrectionFactor = self.__distortionCorrectionFloor * 2
      except:
        return
    return

  def __orderPointsByY(self, distortionLinePoints: List[List[float]]):
    relevantPoints = distortionLinePoints[:3]
    relevantPoints.sort(key=lambda p: p[0])
    left = relevantPoints[0]
    middle = relevantPoints[1]
    right = relevantPoints[2]
    return left, middle, right

  def __findOffsetToLineForUndistortedPoints(self, left: List[float], middle: List[float], right: List[float]) -> float:
    leftUndistorted = self.correctDistortionForPoint(left)
    middleUndistorted = self.correctDistortionForPoint(middle)
    rightUndistorted = self.correctDistortionForPoint(right)
    b = (rightUndistorted[1] - leftUndistorted[1])/(rightUndistorted[0] - leftUndistorted[0])
    a = leftUndistorted[1] - b*leftUndistorted[0]
    expectedMiddleY = a + b*middleUndistorted[0]
    middleYError = expectedMiddleY - middleUndistorted[1]
    return middleYError

  def __findUndistortedPerspectiveShift(self, startpoints: List[List[float]], endpoints: List[List[float]]) -> None:
    undistortedStartpoints = [self.correctDistortionForPoint(startpoint) for startpoint in startpoints]
    self.__perspectiveShiftCoefficients = self.__findPerspectiveShiftCoefficients(undistortedStartpoints, endpoints)

  def __findPerspectiveShiftCoefficients(self, startpoints: List[List[float]], endpoints: List[List[float]]) -> List[float]:
    # Taken from the pytorch perspective transform. Slight changes from original.
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)
    for i, (p1, p2) in enumerate(zip(startpoints, endpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    b_matrix = torch.tensor(endpoints, dtype=torch.float).view(8)
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution
    output: List[float] = res.tolist()
    return output