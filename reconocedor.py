import cv2
import numpy
import os
import argparse
import math
import keras
from keras.models import load_model
import decimal
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID";
os.environ['CUDA_VISIBLE_DEVICES']="1";

decimal.getcontext().rounding = decimal.ROUND_DOWN

def nms(boxes, overlapTh):
    if len(boxes) == 0:
        return []
    seleccionados = []
    
    xini = numpy.array([i[0] for i in boxes]) #coordenadas x de los puntos iniciales
    yini = numpy.array([i[1] for i in boxes]) #coordenadad y de los puntos iniciales
    xfin = numpy.array([i[2] for i in boxes]) #coordenadas x de los puntos finales
    yfin = numpy.array([i[3] for i in boxes]) #coordenadas y de los puntos finales
    scores = numpy.array([i[4] for i in boxes]) #score de las cajas

    area = (xfin - xini + 1) * (yfin - yini + 1) #calculo vectorial (speedup) del area de las bboxes
    indices = numpy.argsort(yfin) #orden de las bboxes por valor de y en su punto final

    while len(indices) > 0: #mientras queden bboxes por analizar
    
        ultimo = len(indices) - 1   
        i = indices[ultimo]
        seleccionados.append(i)

        xini_max = numpy.maximum(xini[i], xini[indices[:ultimo]])
        yini_max = numpy.maximum(yini[i], yini[indices[:ultimo]])
        xfin_min = numpy.minimum(xfin[i], xfin[indices[:ultimo]])
        yfin_min = numpy.minimum(yfin[i], yfin[indices[:ultimo]])

        w = numpy.maximum(0, xfin_min - xini_max + 1)
        h = numpy.maximum(0, yfin_min - yini_max + 1)
        # compute the ratio of overlap

        overlap = (w * h) / area[indices[:ultimo]]

        indices = numpy.delete(indices, numpy.concatenate(([ultimo],
            numpy.where(overlap > overlapTh)[0])))

    ret_list = [boxes[i] for i in seleccionados]
    return ret_list

def pad(img):
    desired_size = 21

    old_size = img.shape[:2] # old_size is in (height, width) format

    try:
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
    except:
        return None

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

def main():
    
    parser = argparse.ArgumentParser(description="Parsing command line argunments...")
    parser.add_argument("--img", default=None, type=str, required=False, help="Route to image file")
    parser.add_argument("--folder", default=None, type=str, required=False, help="Route to input file")
    parser.add_argument("--thr", default=0.999, type=float, required=False, help="Face detection threshold")

    args = parser.parse_args()
    thr = args.thr
    images = []
    if args.folder is None:
        images.append(args.img)
    else:
        print(str(os.listdir(args.folder)))
        images = [os.path.join(args.folder, x) for x in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder,x))]
        print(str(images))

    count = 0
    for i in images:
        print("Analising image: " + i)
        imgOriginal = cv2.imread(i)
        if imgOriginal is None:
            continue
        hOr, wOr, cOr = imgOriginal.shape
        crops = {}
        prev = min(hOr, wOr)
        times = 0
        step = 21
        while prev > step:
            key = math.pow(0.9,times)
            times = times + 1
            prev = int(prev*0.9)
            value = []
            newSize = (math.floor(hOr*key), math.floor(wOr*key))
            img = cv2.resize(imgOriginal, newSize, interpolation=cv2.INTER_CUBIC)
            print(img.shape)
            print("PREV: " + str(prev))
            samples = []
            positions = []
            h, w, c = img.shape
            for y in range(0,w,2):
                for x in range(0, h, 2):
                    aux = None
                    if x+step <= h and y+step <= w: 
                        crop_img = img[x:x+step, y:y+step]
                        aux = (math.floor(x/key), math.floor(y/key), math.floor((x+step)/key), math.floor((y+step)/key))
                    elif x+step > h and y+step > w:
                        crop_img = img[x:w, y:h]
                        crop_img = pad(crop_img)
                        aux = (math.floor(x/key), math.floor(y/key), math.floor(w/key), math.floor(h/key))
                    elif x+step > h:
                        crop_img = img[x:w, y:y+step]
                        crop_img = pad(crop_img)
                        aux = (math.floor(x/key), math.floor(y/key), math.floor(w/key), math.floor((y+step)/key))
                    else:
                        crop_img = img[x:x+step, y:h]
                        crop_img = pad(crop_img)
                        aux = (math.floor(x/key), math.floor(y/key), math.floor((x+step)/key), math.floor(h/key))

                    if crop_img is not None and crop_img.shape == (step,step,3) and aux is not None:
                        #crop_img = crop_img / 255
                        samples.append(crop_img)
                        positions.append(aux)
                        count = count + 1

            samples = numpy.asarray(samples) / 255
            value.append(samples)
            value.append(positions)
            crops[key] = value

        model = load_model('model_faces_20_iter6_dev_DA.hdf5')
        boxes = []
        count = 0
        for key, value in crops.items():
            flags = model.predict_proba(value[0], batch_size=10000)
            print(flags[0])
            print(str(type(flags[0])))
            for i in range(len(flags)):
                if flags[i][1] >= args.thr:
                    boxes.append(tuple(list(value[1][i]) + [flags[i][1]]))
                    print("IS A FACE: " + str(flags[i]))
                    count = count + 1
    
        sup_boxes = nms(boxes, 0.3)
                       
        for (startX, startY, endX, endY, score) in boxes:
            cv2.rectangle(imgOriginal, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        for (startX, startY, endX, endY, score) in sup_boxes:
            cv2.rectangle(imgOriginal, (startX, startY), (endX, endY), (0, 255, 0), 3)
        cv2.imwrite("result.png", imgOriginal)
    print("Last counter: " + str(count))



main()

                
                
