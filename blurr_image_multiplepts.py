import sys
import cv2 as cv
import numpy as np
#from xml.dom import minidom
from bs4 import BeautifulSoup
#import glob
from os import listdir
from os.path import isfile, join
from ast import literal_eval

class Blurr_image_multiplepts:
    """ Author: Emmanuel Harbour
        Date: 11/03/2020
        Description: This class takes as input labels, file path of images to be blurred, output path of blurred images, xml file path
                        and then blurs all images based on the labels given. The mode of blurring is done using GaussianBlur for box
                        and polygon points.
    """

    def __init__(self, image_dir, blurred_image_dir, xmlfile, blur_labels):
        self.image_dir = image_dir
        self.blurred_image_dir = blurred_image_dir
        self.xmlfile = xmlfile
        self.blur_labels = blur_labels

    def get_all_images_name(self, rimage_dir):

        onlyfiles = [f for f in listdir(rimage_dir) if isfile(join(rimage_dir, f))]
        
        return onlyfiles

    def progressbar(self, count, total, suffix=''):
        bar_len = 20
        filled_len = int(round(bar_len * count / float(total)))

        #percents = round(100.0 * count / float(total), 1)
        percents = round(100 * count / float(total))
        bar = '*' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()

    def ug_blur(self, image, image_pts):

        blurred_image = cv.GaussianBlur(image,(81, 81), 30)

        roi_corners = np.array(np.matrix(image_pts))

        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count

        roi_corners = [roi_corners.astype(int)]

        cv.fillPoly(mask, roi_corners, ignore_mask_color)

        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
        final_image = cv.bitwise_and(blurred_image, mask) + cv.bitwise_and(image, mask_inverse)

        return final_image

    def getpoints_from_box(self, prop_box, label):

        image_pts = []

        if prop_box.get("label") == label:

            xtl = prop_box.get("xtl")
            ytl = prop_box.get("ytl")
            xbr = prop_box.get("xbr")                
            ybr = prop_box.get("ybr")                

            image_pts.append((xtl, ytl, xbr, ybr)) 

        if not image_pts:
            return None
        
        return image_pts

    def getpoints_from_polygons(self, prop_polygon, label):
        image_pts = []

        if prop_polygon.get("label") == label:

            image_pts.append(prop_polygon.get('points'))

        if not image_pts:
            return None

        return image_pts

    def find_label(self, soup, file, blur_labels):

        image_pts = dict.fromkeys(["box", "polygon"])

        blur_prop = soup.find("image", {"name":file})
        
        if blur_prop is None:
            return image_pts

        prop_boxes = blur_prop.find_all('box')
        prop_polygons = blur_prop.find_all('polygon')
        
        #get box points and polygon points for each label given
        for label in blur_labels:
            for prop_box in prop_boxes:

                bxpts = self.getpoints_from_box(prop_box, label)

                if bxpts is not None:
                    lst_bpts = image_pts['box']

                    if lst_bpts is None:
                        image_pts['box'] = [str(bxpts).strip('[]')]

                    else:
                        lst_bpts.append(str(bxpts).strip('[]'))
                        image_pts['box'] = lst_bpts

            for prop_polygon in prop_polygons:
                polypts = self.getpoints_from_polygons(prop_polygon, label)

                if polypts is not None:
                    if image_pts['polygon'] is None:
                        image_pts['polygon'] = polypts
                    #elif len(p) > 1:
                    #    image_pts['polygon'].extend(p)
                    else:
                        [image_pts['polygon'].append(' '.join(map(str, polypts)))]


        #print(image_pts)
        return image_pts

    def get_blur_points(self, soup, file, blur_labels): #pts_list all points in array list to be blurred.
        #dict_g = []

        dict_g = self.find_label(soup, file, blur_labels)

        #print(dict_g)
        return dict_g

    def blur_all_boxes(self, image, box_pts):

        h, w = image.shape[:2]
        # gaussian blur kernel size depends on width and height of original image
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1

        for lp in box_pts:

            #print(type(lp))
            start_x, start_y, end_x, end_y = lp

            start_x = round(float(start_x))
            start_y = round(float(start_y))
            end_x = round(float(end_x))
            end_y = round(float(end_y))
            #print(start_x, start_y, end_x, end_y)
    
            blur_pt = image[start_y: end_y, start_x: end_x]
            # apply gaussian blur to this blur_pt
            blur_pt = cv.GaussianBlur(blur_pt, (kernel_width, kernel_height), 0)
            # put the blurred blur_pt into the original image
            image[start_y: end_y, start_x: end_x] = blur_pt

        return image

    def blur_all_polygons(self, image, polygon_pts):
        #blurred_image = cv.GaussianBlur(image,(81, 81), 30)
        #p_image = image.copy

        for image_pts in polygon_pts:
            blurred_image = cv.GaussianBlur(image,(81, 81), 30)
            #May loop for poly
            roi_corners = np.array(np.matrix(image_pts))
            mask = np.zeros(image.shape, dtype=np.uint8)
            channel_count = image.shape[2]
            ignore_mask_color = (255,)*channel_count
            roi_corners = [roi_corners.astype(int)]

            cv.fillPoly(mask, roi_corners, ignore_mask_color)
            mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask

            image = cv.bitwise_and(blurred_image, mask) + cv.bitwise_and(image, mask_inverse)

        return image

    def box_blur(self, image, box_pts):

        blur_box_img = self.blur_all_boxes(image, box_pts)

        return blur_box_img

    def poly_blur(self, image, polygon_pts):

        blur_polygon_img = self.blur_all_polygons(image, polygon_pts)

        return blur_polygon_img

    def start(self):
        rawimages = self.get_all_images_name(self.image_dir)
        xmlattr= self.xmlfile

        infile = open(xmlattr,"r")            
        contents = infile.read()
        soup = BeautifulSoup(contents, 'xml')

        print(len(rawimages), "total images to be blurred.")
        counter = 0

        for file in rawimages:

            final_image = None
            image = cv.imread(join(self.image_dir, file))

            image_pts = self.get_blur_points(soup, file, self.blur_labels)

            #blur boxes.
            if image_pts['box'] is not None:
                bxpts = image_pts['box']

                if len(bxpts) > 1:
                    bxpts = list(literal_eval(','.join(bxpts)))
                else:
                    bxpts = list(map(eval, bxpts))

                image = self.box_blur(image, bxpts)

            #blur polygons
            if image_pts['polygon'] is not None:
                polygon_pts = image_pts['polygon']

                image = self.poly_blur(image, polygon_pts)

            if image_pts['box'] is not None or image_pts['polygon'] is not None:
                
                cv.imwrite(join(self.blurred_image_dir, file), image)

            else:
                cv.imwrite(join(self.blurred_image_dir, file), image)

            counter += 1
            self.progressbar(counter, len(rawimages), "[{} of {} total images blurred]".format(counter, len(rawimages)))
        print("\ncompleted")



