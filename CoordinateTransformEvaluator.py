import numpy as np
import zipfile
import sys
import cv2
from typing import List
from TransformCalibrator import TransformCalibrator
from ImageCoordinateTransformEvaluator import ImageCoordinateTransformEvaluator

def rescaleImage(image, scale):
  width = int(image.shape[1] * scale)
  height = int(image.shape[0] * scale)
  dim = (width, height)
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return resized

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
   
def makeMapInfo(scale = 1):
  referenceMap = readImageFromZip(archive, "extractions/ReferenceMap.png")
  rescaledReferenceMap = rescaleImage(referenceMap, 0.5)
  mapPoint1 = {"x": 40, "y": 25, "label": "1"}
  mapPoint2 = {"x": 805, "y": 25, "label": "2"}
  mapPoint3 = {"x": 210, "y": 178, "label": "3"}
  mapPoint4 = {"x": 725, "y": 300, "label": "4"}
  mapPoint5 = {"x": 422, "y": 25, "label": "5"}
  mapPoints = [mapPoint1, mapPoint2, mapPoint3, mapPoint4, mapPoint5]
  return rescaledReferenceMap, mapPoints

def processFolder(zipArchive, folderPath, referenceMap, mapPoints):
  print("Processing folder: " + folderPath)
  filesInFolder = listFilesInZippedFolder(zipArchive, folderPath)
  calibrationFileName = folderPath + "calibration.json"
  if calibrationFileName in filesInFolder:
    print("Folder already has configuration.json. Skipping.")
    return
  if folderPath + "background.jpg" not in filesInFolder:
    print("Folder has no background file. Skipping.")
    return
  background = readImageFromZip(zipArchive, folderPath + "background.jpg")
  calibrator = TransformCalibrator(referenceMap, background, scale=0.5)
  transform, blockBoxes = calibrator.run(mapPoints,["1", "2", "3", "4"], ["1", "2", "5"])
  fileForProcessing = filterImageFilesForProcessing(filesInFolder)
  for filePath in fileForProcessing:
    evaluateImage(zipArchive, filePath, referenceMap, transform, blockBoxes)
  cv2.destroyAllWindows()
  calibrator.saveCalibration(zipArchive, folderPath, transform, blockBoxes)
  
def evaluateImage(zipArchive, filePath, referenceMap, transform, blockBoxes):
  print("Processing file: " + filePath)
  imageTransformEvaluator = ImageCoordinateTransformEvaluator(metersPerPixel, frameScale = 0.5)
  imageTransformEvaluator.evaluateImage(referenceMap, zipArchive, filePath, transform, blockBoxes)

fieldLengthInMeters = 16.46
fieldLengthInPixels = 858
metersPerPixel = fieldLengthInMeters / fieldLengthInPixels

zipPath = sys.argv[1]
with zipfile.ZipFile(zipPath, "a") as archive:
  rescaledReferenceMap, mapPoints = makeMapInfo(scale = 0.5)
  folders = listFoldersInZipArchive(archive)
  for folder in folders:
    processFolder(archive, folder, rescaledReferenceMap, mapPoints)

cv2.destroyAllWindows()