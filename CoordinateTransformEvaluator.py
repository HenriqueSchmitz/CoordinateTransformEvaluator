import numpy as np
import zipfile
import sys
import cv2
from typing import List
from TransformCalibrator import TransformCalibrator
from TensorflowImageDetector import TensorflowImageDetector

def rescaleImage(image, scalePercentage):
  width = int(image.shape[1] * scalePercentage / 100)
  height = int(image.shape[0] * scalePercentage / 100)
  dim = (width, height)
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return resized

def readImageFromZip(zip, imagePath):
    imageData = zip.read(imagePath)
    image = cv2.imdecode(np.frombuffer(imageData, np.uint8), 1)
    return image

rescaled = None

zipPath = sys.argv[1]
archive = zipfile.ZipFile(zipPath, "r")
folders = [file for file in archive.namelist() if file.endswith('/')]
for folder in folders:
    print(folder)

print()
print("=============================================")
print()

testFolder = folders[1]
print(testFolder)
filesInTestFolder = [file for file in archive.namelist() if file.startswith(testFolder)]
relevantFiles = [file for file in filesInTestFolder if file.endswith(".jpg") and not file.endswith("background.jpg")]
for file in relevantFiles:
  print(file)

referenceMap = readImageFromZip(archive, "extractions/ReferenceMap.png")
rescaledReferenceMap = rescaleImage(referenceMap, 50)

image = readImageFromZip(archive, testFolder + "background.jpg")
rescaled = rescaleImage(image, 50)
calibrator = TransformCalibrator(rescaledReferenceMap, image)
mapPoint1 = {"x": 40, "y": 25, "label": "1"}
mapPoint2 = {"x": 805, "y": 25, "label": "2"}
mapPoint3 = {"x": 210, "y": 178, "label": "3"}
mapPoint4 = {"x": 725, "y": 300, "label": "4"}
mapPoint5 = {"x": 422, "y": 25, "label": "5"}
mapPoints = [mapPoint1, mapPoint2, mapPoint3, mapPoint4, mapPoint5]
transform, blockBoxes = calibrator.run(mapPoints,["1", "2", "3", "4"], ["1", "2", "5"])

imagePath = relevantFiles[0]
testedImage = readImageFromZip(archive,imagePath)
detector = TensorflowImageDetector()
detections = detector.loadDetectionsFromZip(archive, imagePath.replace(".jpg", ".json"))
imageWithDetections = detector.drawDetectionsOnImage(testedImage, detections, ["robot"], blockBoxes, drawMidpoint = True, yCenterFromTopLeft=0.67)
rescaledTestedImage = rescaleImage(imageWithDetections, 50)
cv2.imshow("test", rescaledTestedImage)
cv2.waitKey(0)

cv2.destroyAllWindows()