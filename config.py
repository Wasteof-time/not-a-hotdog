SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42
CLASSES = ["hot_dog", "not_hot_dog"]  # ImageFolder order: hot_dog=0, not_hot_dog=1
DATA_DIR = "data"
WEIGHTS = "weights/seefood.pt"
CALIB = "weights/calibration.json"
OUT_DIR = "outputs"
