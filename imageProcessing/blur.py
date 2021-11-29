from PIL import Image, ImageFilter
import os

images = [f for f in os.listdir('yalefacesmod') if f != ".DS_Store"]
X = [Image.open("./yalefacesmod/" + image) for image in images]
for val in range(len(X)):
    blurImage = X[val].filter(ImageFilter.BLUR)
    blurImage.save("./yalefacesmod/" + "blur" + images[val])

