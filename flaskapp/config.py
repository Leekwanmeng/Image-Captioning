import os 

class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess-it"
    UPLOADED_PHOTOS_DEST = os.path.join(os.getcwd(), "uploads/")
    MODELS_FOLDER = os.path.join(os.getcwd(), "dl_models/")
    DATA_FOLDER = os.path.join(os.getcwd(), "dl_models/")