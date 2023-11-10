import statistics
import zipfile
import json
import math
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
  identificationEvaluationFiles = [file for file in files if file.endswith("evaluation_points.json")]
  pointPairs = []
  for relevantFile in identificationEvaluationFiles:
    fileEvaluation = loadDetections(archive, relevantFile)
    pointPairs.extend(fileEvaluation["pointPairs"])
  pointDistances = [pointPair["measurementDistance"] for pointPair in pointPairs]
  numberOfEvaluations = len(pointPairs)
  print("Robots evaluated: " + str(numberOfEvaluations))
  print("Average distance: " + str(sum(pointDistances) / numberOfEvaluations))
  standardDeviation = statistics.pstdev(pointDistances)
  print("Standard deviation: " + str(standardDeviation))
  Z_95_percent = 1.98
  errorAt95PercentConfidence = Z_95_percent * standardDeviation / math.sqrt(numberOfEvaluations)
  print("Error at 95%% confidence: " + str(errorAt95PercentConfidence))