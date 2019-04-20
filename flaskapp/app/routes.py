from flask import render_template, flash, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug import secure_filename
import torchvision.transforms as transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import skimage.transform

from app import app
from app.settings import MODELS_FOLDER, DATA_FOLDER
from app.forms import ImageForm

import os
import time
import sys

sys.path.append('../')
from test_sampler_mok import model


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

@app.route('/')
@app.route('/index')
@app.route('/upload/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    if form.validate_on_submit():
        # Remove cached image files
        file_path = os.path.join('uploads', 'uploaded_image.jpg')
        if os.path.exists(file_path):
            os.remove(file_path)
        filename = photos.save(form.upload.data, name="uploaded_image.jpg")
        file_url = photos.url(filename) + f"?{time.time()}"
        caption = get_caption_and_masked_images('uploads/' + filename)
        if os.path.exists(os.path.join("uploads", "attention_plot.jpg")):
            attention_path = photos.url("attention_plot.jpg") + f"?{time.time()}"
        else:
            attention_path = None
    else:
        file_url = None
        caption = None
        attention_path = None
    return render_template(
        'index.html',
        form=form,
        file_url=file_url,
        caption=caption,
        attention=attention_path
    )

def get_caption_and_masked_images(filename):
    sentence, seq, alphas, idx2word = model.caption_image_beam_search(filename)
    alphas = torch.FloatTensor(alphas)
    visualize_att(filename, seq, alphas, idx2word)
    return sentence

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[str(ind)] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    # plt.show()
    plot_path = os.path.join("uploads/", "attention_plot.jpg")
    plt.savefig(plot_path)
    plt.clf()
