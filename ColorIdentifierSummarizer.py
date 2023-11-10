import zipfile
import json
import sys

def loadDetections(zipArchive, filePath):
  readFile = ""
  with zipArchive.open(filePath, 'r') as file:
    readFile = file.read()
  loadedDetections = json.loads(readFile)
  return loadedDetections

print("Loading Evaluations...")
zipPath = sys.argv[1]
with zipfile.ZipFile(zipPath, "r") as archive:
  files = [file for file in archive.namelist() if not file.endswith('/')]
  identificationEvaluationFiles = [file for file in files if file.endswith("identifier_evaluations.json")]
  detections = []
  for relevantFile in identificationEvaluationFiles:
    detectionsForFrame = loadDetections(archive, relevantFile)
    detections.extend(detectionsForFrame)
  successes = [detection for detection in detections if detection["isDetectionCorrect"]]
  print("Frames evaluated: " + str(len(identificationEvaluationFiles)))
  print("Detections evaluated: " + str(len(detections)))
  print("Scuccessful identifications: " + str(len(successes)))
  precision = len(successes) / len(detections)
  print("Precision: " + str(precision))