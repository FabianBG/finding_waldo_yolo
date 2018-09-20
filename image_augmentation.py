import imgaug as ia
from imgaug import augmenters as iaa
import os
import cv2

ia.seed(1)

# box x1 x2 y1 y2
def convert_yolo(size, box):
    dw = 1./size[1]
    dh = 1./size[0]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
# box x1 y1 x2 y2
def convert_axis(size, box):
    dw = 1./size[1]
    dh = 1./size[0]
    x = box[0]/dw
    w = box[2]/dw
    y = box[1]/dh
    h = box[3]/dh

    b1 = x + (w/2)
    b3 = y + (h/2)
    b0 = x + (w/2) - w
    b2 = y + (h/2) - h

    return (b0, b1, b2, b3)



directory_yolo = "/home/mbastidas/git/yolo-models/waldo/data/"
directory_aug = "/home/mbastidas/git/yolo-models/waldo/data/aug/"
image_extension = ".jpg"
bbox_images = []
print(convert_yolo((1840, 3264), (225, 1412, 774, 1794)))
print(convert_axis((1840, 3264),(0.444837, 0.393382, 0.645109, 0.312500)))
print("Cargar secuencia de transformaciones ...")
sometimes = lambda aug: iaa.Sometimes(0.8, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of all images
    iaa.Flipud(0.2), # vertically flip 20% of all images
    # crop some of the images by 0-10% of their height/width
    sometimes(iaa.Crop(percent=(0, 0.1))),
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.7))
    ),
    sometimes(iaa.Affine(
            scale={"x": (0.2, 0.7), "y": (0.2, 0.7)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 1),
            mode=ia.ALL
        )),
     # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
    iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)
iteration = 0
max_ite = 47
init_ite = 0
repeats = 20
print("Leyendo directorio de archivos ...")
for file in os.listdir(directory_yolo):
    if file.endswith(".txt") and file != "classes.txt":
        with open(directory_yolo  + file, "r") as bbox_file:
            print("Cargando imagen", directory_yolo + file.split(".")[0] + image_extension)
            data = {
                    "image": directory_yolo  + file,
                    "array": cv2.imread(directory_yolo + file.split(".")[0] + image_extension),
                    "names": []
                }
            iteration = iteration + 1
            if iteration > max_ite: break
            if iteration < init_ite: 
                print("Saltando iteracion ", iteration)
                continue

            bbox = [] 
            for line in bbox_file:
                line = line.split(" ")
                if len(line) != 5 : continue
                name, x1, y1, x2, y2 = line
                x1, x2, y1, y2 = convert_axis((data["array"].shape),
                (float(x1), float(y1), float(x2), float(y2)))
                data["names"].append(name)
                bbox.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            data["bbox"] = ia.BoundingBoxesOnImage(bbox, shape=data["array"].shape)
            bbox_images.append(data)
            print("Aplicando transformaciones", data["array"].shape)
            seq_det = seq.to_deterministic()
            images_aug = seq_det.augment_images([data["array"]] * repeats)
            bbss_aug = seq_det.augment_bounding_boxes([data["bbox"]] * repeats)
            print("Guardando imagenes aumentadas " + file.split(".")[0] + image_extension + " " +
            str(len(images_aug)))
            for i in range(0, len(images_aug)):
                image_aug = images_aug[i]
                bbs_aug = bbss_aug[i]
                cv2.imwrite(directory_aug + file.split(".")[0] + "_" + str(i) + image_extension, image_aug)
                with open(directory_aug + file.split(".")[0] + "_" + str(i) + ".txt", "w") as yolo_txt:
                    j = 0
                    #bbs_aug = bbs_aug.cut_out_of_image()
                    box = bbs_aug.bounding_boxes
                    for name in data["names"]:
                        if box[j].is_fully_within_image(data["array"].shape):
                            x1, y1, x2, y2 = convert_yolo(data["array"].shape, 
                            (box[j].x1, box[j].x2, box[j].y1 ,box[j].y2))
                            yolo_txt.write("%s %.4f %.4f %.4f %.4f\n" %
                            (name, x1, y1, x2, y2) )
                        j = j + 1
            print("Fin " + file + " itenracion " + str(iteration))
'''
image_after = bbs_aug.remove_out_of_image().cut_out_of_image().draw_on_image(image_aug, thickness=5, color=[0, 0, 255])
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600,600)
                cv2.imshow('image', image_after)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
'''
