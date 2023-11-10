from DistortionAdjustedCoordinateTransform import DistortionAdjustedCoordinateTransform
from matplotlib import pyplot as plt
import statistics
import numpy as np
import zipfile
import json
import math
import sys
import csv

def loadZippedJson(zipArchive, filePath):
  readFile = ""
  with zipArchive.open(filePath, 'r') as file:
    readFile = file.read()
  loadedDetections = json.loads(readFile)
  return loadedDetections

def getBasePoint(detection, robotBaseFromTopPercentage):
  minX = detection["topLeftPoint"][0]
  minY = detection["topLeftPoint"][1]
  maxX = detection["bottomRightPoint"][0]
  maxY = detection["bottomRightPoint"][1]
  x = round(minX + (maxX - minX)/2)
  y = round(minY + (maxY - minY)*robotBaseFromTopPercentage)
  return [x,y]

def calculateDistanceBetweenPoints(pointA, pointB):
  return math.sqrt(math.pow(pointA[0] - pointB[0], 2) + math.pow(pointA[1] - pointB[1], 2))

def getEvaluatedDistancesForDistanceFactor(archive, files, calibrationFiles, distanceFactor):
  pointDistances = []
  for calibrationFile in calibrationFiles:
    calibrationData = loadZippedJson(archive, calibrationFile)
    transformParams = calibrationData["transformParameters"]
    transform = DistortionAdjustedCoordinateTransform(transformParams["distortionLinePoints"], transformParams["distortionCenter"], transformParams["seenReferencePoints"], transformParams["targetReferencePoints"])
    calibrationFileFolder = calibrationFile.replace("calibration.json", "")
    evaluationFiles = [file for file in files if file.startswith(calibrationFileFolder) and file.endswith("evaluation_points.json")]
    for evaluationFile in evaluationFiles:
      evaluationData = loadZippedJson(archive, evaluationFile)
      originalDetections = [pointPair["detectedPoint"] for pointPair in evaluationData["pointPairs"]]
      userProvidedPoints = evaluationData["userProvidedPoints"]
      for originalDetection in originalDetections:
        newBasePoint = getBasePoint(originalDetection, distanceFactor)
        mappedNewPoint = transform.roundedShiftPerspectiveForPoint(newBasePoint)
        minimumDistance = None
        for userProvidedPoint in userProvidedPoints:
          distance = calculateDistanceBetweenPoints(mappedNewPoint, userProvidedPoint)
          if minimumDistance == None or distance < minimumDistance:
            minimumDistance = distance
        distanceInMeters = minimumDistance * metersPerPixel
        pointDistances.append(distanceInMeters)
  return pointDistances

fieldLengthInMeters = 16.46
fieldLengthInPixels = 858
metersPerPixel = fieldLengthInMeters / fieldLengthInPixels

print("Loading Evaluations...")
zipPath = sys.argv[1]
with zipfile.ZipFile(zipPath, "r") as archive:
  files = [file for file in archive.namelist() if not file.endswith('/')]
  calibrationFiles = [file for file in files if file.endswith("calibration.json")]
  heightDistances = []
  # for robotBaseFromTopPercentage in range(50, 100):
  #   distanceFactor = robotBaseFromTopPercentage/100
  #   pointDistances = getEvaluatedDistancesForDistanceFactor(archive, files, calibrationFiles, distanceFactor)
  #   numberOfEvaluations = len(pointDistances)
  #   averageDistance = sum(pointDistances) / numberOfEvaluations
  #   print("Robots evaluated: " + str(numberOfEvaluations))
  #   print("Average distance: " + str(averageDistance))
  #   heightDistances.append([distanceFactor, averageDistance])
  # heightDistancesConverted = [[str(round(heightDistance[0],2)), str(round(heightDistance[1],4))] for heightDistance in heightDistances]
  # print(heightDistancesConverted)
  # with open("heightDistances.csv", "w") as file:
  #   writer = csv.writer(file)
  #   writer.writerows(heightDistancesConverted)
  pointDistances = getEvaluatedDistancesForDistanceFactor(archive, files, calibrationFiles, 0.78)
  # pointDistances = getEvaluatedDistancesForDistanceFactor(archive, files, calibrationFiles, 0.67)
  numberOfEvaluations = len(pointDistances)
  averageDistance = sum(pointDistances) / numberOfEvaluations
  print("Robots evaluated: " + str(numberOfEvaluations))
  print("Average distance: " + str(averageDistance))
  medianDistance = statistics.median(pointDistances)
  print("Median distance: " + str(medianDistance))
  standardDeviation = statistics.pstdev(pointDistances)
  print("Standard deviation: " + str(standardDeviation))
  Z_95_percent = 1.98
  errorAt95PercentConfidence = Z_95_percent * standardDeviation / math.sqrt(numberOfEvaluations)
  print("Error at 95%% confidence: " + str(errorAt95PercentConfidence))

  data = np.array(pointDistances) 
  bins = np.linspace(min(data), 
                    max(data),
                    30) # fixed number of bins
  plt.xlim([0, max(data)+0.5])
  plt.hist(data, bins=bins, alpha=0.5)
  plt.title("Evaluation of robot position estimator with base 78% of the way down the detection")
  plt.xlabel("Distance between human provided robot position and estimated robot position in meters")
  plt.ylabel("Count")
  plt.show()