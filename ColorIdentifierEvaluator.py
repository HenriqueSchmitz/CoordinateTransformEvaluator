from TensorflowImageDetector import TensorflowImageDetector
from RobotIdentifier import RobotIdentifier
import numpy as np
import zipfile
import json
import sys
import cv2

def readImageFromZip(zip, imagePath):
  imageData = zip.read(imagePath)
  image = cv2.imdecode(np.frombuffer(imageData, np.uint8), 1)
  return image

def listFoldersInZipArchive(zipArchive):
  return [path for path in zipArchive.namelist() if path.endswith('/')] 

def listFilesInZippedFolder(zipArchive, folerPath):
  return [file for file in zipArchive.namelist() if file.startswith(folerPath)]

def filterImageFilesForProcessing(listOfFiles):
  return [file for file in listOfFiles if file.endswith(".jpg") and not file.endswith("background.jpg")]

def readImageFromZip(zip, imagePath):
  imageData = zip.read(imagePath)
  image = cv2.imdecode(np.frombuffer(imageData, np.uint8), 1)
  return image

def cropDetection(image, detection):
    topLeftPoint = detection["topLeftPoint"]
    bottomRightPoint = detection["bottomRightPoint"]
    xMin = topLeftPoint[0]
    yMin = topLeftPoint[1]
    xMax = bottomRightPoint[0]
    yMax = bottomRightPoint[1]
    return image[yMin:yMax, xMin:xMax].copy()

def generateFeedbackSquare(size, color):
  adjustedColor = np.float64(color)
  full = np.full((size, size, 3), adjustedColor)
  return full

feedbackSquareSize = 150
red = (0, 0, 255)
blue = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
purple = (255, 0, 255)
yellow = (255, 255, 0)
enterCode = 13

def updateFeedbackSquare(color):
  feedbackSquare = generateFeedbackSquare(feedbackSquareSize, color)
  cv2.imshow("Feedback", feedbackSquare)

def saveEvaluationsForImage(zipArchive, imagePath, evaluations):
  evaluationResultsJson = json.dumps(evaluations)
  evaluationsResultPath = buildEvaluationFilePath(imagePath)
  with zipArchive.open(evaluationsResultPath, 'w') as file:
    file.write(evaluationResultsJson.encode())
    print("Saved evaluation points: " + evaluationsResultPath)

def buildEvaluationFilePath(imagePath):
  return imagePath.replace(".jpg", "_identifier_evaluations.json")

def processDetection(frame, detection):
  selecting = True
  detectionImage = cropDetection(frame, detection)
  cv2.imshow("Detection", detectionImage)
  updateFeedbackSquare(black)
  action = "select"
  userProvidedColor = None
  while selecting:
    key = cv2.waitKey(0)
    if key == ord("q"):
      updateFeedbackSquare(yellow)
      action = "quit"
    elif key == ord("a"):
      updateFeedbackSquare(blue)
      action = "blue"
    elif key == ord("d"):
      updateFeedbackSquare(red)
      action = "red"
    elif key == ord("w"):
      updateFeedbackSquare(white)
      action = "unkown"
    elif key == ord("s"):
      updateFeedbackSquare(purple)
      action = "skip"
    elif key == enterCode:
      if action == "quit":
        exit()
      elif action == "skip":
        return None
      elif action == "unkown":
        userProvidedColor = "unkown"
        selecting = False
      elif action == "blue":
        userProvidedColor = "blue"
        selecting = False
      elif action == "red":
        userProvidedColor = "red"
        selecting = False
  detectedColor = robotIdentifier.identifyRobot(frame, detection)
  detectionEvaluation = {}
  detectionEvaluation["userProvidedColor"] = userProvidedColor
  detectionEvaluation["detectedColor"] = detectedColor
  detectionEvaluation["isDetectionCorrect"] = userProvidedColor == detectedColor
  detectionEvaluation["topLeftPoint"] = detection["topLeftPoint"]
  detectionEvaluation["bottomRightPoint"] = detection["bottomRightPoint"]
  cv2.destroyWindow("Detection")
  return detectionEvaluation

robotIdentifier = RobotIdentifier()
zipPath = sys.argv[1]
with zipfile.ZipFile(zipPath, "a") as archive:
  folders = listFoldersInZipArchive(archive)
  for folder in folders:
    print("Evaluating match: " + folder)
    filesInFolder = listFilesInZippedFolder(archive, folder)
    fileForProcessing = filterImageFilesForProcessing(filesInFolder)
    for filePath in fileForProcessing:
      print("Evaluating frame: " + filePath)
      evaluationFileName = buildEvaluationFilePath(filePath)
      if evaluationFileName in filesInFolder:
        print("Image has already been evaluated for identification. Skipping")
        continue
      frame = readImageFromZip(archive, filePath)
      detector = TensorflowImageDetector()
      detections = detector.loadDetectionsFromZip(archive, filePath.replace(".jpg", ".json"))
      relevantDetections = [detection for detection in detections if detection["label"] == "robot"]
      detectionEvaluations = []
      for detection in relevantDetections:
        detectionEvaluation = processDetection(frame, detection)
        if detectionEvaluation is not None:
          detectionEvaluations.append(detectionEvaluation)
      saveEvaluationsForImage(archive, filePath, detectionEvaluations)