from google_images_download import google_images_download
from os import rename, listdir, remove
from PIL import Image, ImageDraw
import math
import random

download_dir = "./downloads"
out_dir = "./images"
out_size = (150,150)

#Downloads images
print("Downloading...\n\n")
response = google_images_download.googleimagesdownload()
config = {"keywords":"family photo",
            "limit":5,
            "aspect_ratio":"square",
            "type":"photo",
            "output_directory":download_dir,
            "no_directory":True,
            "format":"jpg",
            "chromedriver":"/usr/bin/chromedriver"}
absolute_image_paths = response.download(config)

#Renames
print("Renaming...\n\n")
i = 1
for filename in listdir(download_dir):
    name = download_dir+"/"+"%04d.jpg" % i
    filepath = download_dir+"/"+filename
    rename(filepath, name)
    i = i+1

#Resizes
print("Resizing...\n\n")
for filename in listdir(download_dir):
    try:
        filepath = download_dir+"/"+filename
        image = Image.open(filepath)
        image = image.resize((150, 150), Image.LANCZOS)         
        image.save(filepath)
    except OSError:
        filepath = download_dir+"/"+filename
        remove(filepath)
        pass

#Draws shapes (lines, rectangles or ellipses) on images to generate dataset
print("Fuuck11ing things uppp yoooooooo...\n\n")
times = 3
min_shapes = 2
max_shapes = 4
#Line vars
min_line_size = 15
max_line_size = 150 
min_line_width = 4
max_line_width = 10
#Rectangle vars
min_rec_size = 15
max_rec_size = 30
#ellipse vars
min_ellipse_height = 15
max_ellipse_height = 30
min_ellipse_width = 15
max_ellipse_width = 30
for filename in listdir(download_dir):
    filepath = download_dir+"/"+filename
    for i in range (0, times):
        image = Image.open(filepath)
        draw = ImageDraw.Draw(image)
        num_shapes = random.randint(min_shapes, max_shapes)
        for k in range(0, num_shapes):
            shape = random.randint(0,1)
            #Lines
            if shape == 0:
                line_width = random.randint(min_line_width, max_line_width)
                line_size = random.randint(min_line_size, max_line_size)
                line_angle = random.uniform(0, 2*math.pi)
                #FIXME use image size instead of 150 in randint
                line_x1 = random.randint(1,150)
                line_y1 = random.randint(1,150)
                line_x2 = line_x1 + line_size*math.cos(line_angle)
                line_y2 = line_y1 + line_size*math.sin(line_angle)
                draw.line(( line_x1, 
                            line_y1,
                            line_x2,
                            line_y2), 
                            (255, 255, 255),
                            random.randint(min_line_width, max_line_width))
            #ellipses
            if shape == 1:
                ellipse_width = random.randint(min_ellipse_width, max_ellipse_width)
                ellipse_height = random.randint(min_ellipse_height, max_ellipse_height)
                #FIXME use image size instead of 150 in randint
                ellipse_x1 = random.randint(1,150)
                ellipse_y1 = random.randint(1,150)
                ellipse_x2 = ellipse_x1 + ellipse_width
                ellipse_y2 = ellipse_y1 + ellipse_height
                draw.ellipse((ellipse_x1,
                                ellipse_y1,
                                ellipse_x2,
                                ellipse_y2),
                                (255, 255, 255))

        image.save(out_dir+"/"+filename+"_"+str(i), "JPEG")

         


    

