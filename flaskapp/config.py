import os 

class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess-it"
    UPLOADED_PHOTOS_DEST = os.path.join(os.getcwd(), "uploads/")
