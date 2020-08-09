#!/usr/bin/env python
''' 
this is intended to be used as a standalone script to count and identify cells, or as a library. 
 
'''
import numpy as np
import argparse
import cv2
import os
import sys
import sklearn
import sklearn.pipeline

##
# CV2 Estimators and Transformers


class ThresholdCellArea(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    ''' returns total area covered by cells in image_gray, 
    using an adaptive thresholding and summing method
    cellColor: default of 255 for light cells on dark background, 0 for dark on light bg
    d: 
    ''' 

    def __init__(self, cellColor=255, d=9, color_sigma=30, area_sigma=30, block_size=15,C=0):
        self.cellColor = cellColor 
        self.count = 'undetected'

    def count_cells(self,image):
        ''' image should be of type CellCountImage'''
        im_with_keypoints = self.fit_transform(image.image)
        image.count = self.count
        image.transformed = im_with_keypoints
        self.count = 'undetected'
        
    def fit(self, image_gray):
        ''' takes in a grayscale image, does an opening, returns blob keypoints '''
        #invert if needed

        #adaptive binarization
        blur = cv2.bilateralFilter(image_gray,self.d,self.sigmaColor,self.sigmaSpace)
        if self.cellColor == 0:
            blur = (255-blur)
        elif self.cellColor == 255:
            pass
        else:
            raise ValueError('allowed values for cellColor are 0 and 255, {} was input'.format(self.cellColor))

        thresholded = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0)
    
        white_count = cv2.countNonZero(thresholded)
        #replace with some kind of regression
        
        cell_count = white_count / self.cell_size


class SimpleBlobCellDetector(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    ''' identify cells in image_gray, using simple blob dectection, count and mark them
    default kernel_width determined by parameter sweep''' 

    def __init__(self, kernel_width=5, blobColor=255):
        self.kernel_width = kernel_width
        # default of 255 for light blobs on dark background, 0 for dark on light bg
        self.blobColor = blobColor
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.blobColor = self.blobColor
        self.blob_params.filterByConvexity = False
        self.blob_params.filterByInertia = False
        self.blob_params.minDistBetweenBlobs = 0
        self.blob_params.maxThreshold = 31
        self.blob_params.minThreshold = 0
        self.blob_params.thresholdStep = 0.1
        self.count = 'undetected'

    def simple_blob_cell_detection(self,image):
        ''' image should be of type CellCountImage'''
        im_with_keypoints = self.fit_transform(image.image)
        image.count = self.count
        image.transformed = im_with_keypoints
        self.count = 'undetected'
        
    def fit_transform(self, image_gray):
        ''' takes in a grayscale image, does an opening, returns blob keypoints '''
    
        #blur = cv.GaussianBlur(img,(5,5),0)                            
        #thresholded , distribution  = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        blur = cv2.bilateralFilter(image_gray,9,30,30)
        detector = cv2.SimpleBlobDetector_create(self.blob_params)
        keypoints = detector.detect(blur)

        blue = (0,0,255)
        im_with_keypoints = cv2.drawKeypoints(image_gray, 
                keypoints, np.array([]), blue, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.count = len(keypoints)
        return im_with_keypoints

##
# utility functions

def load_images(args, image_filenames):
    ''' loads images into grayscale numpy.ndarrays'''
    img_out = []
    for image_fname in image_filenames:
        image_orig = CellCountImage(args,image_fname)
        img_out.append(image_orig)
    return img_out

def write_images(images):
    ''' takes in a list of ImageData type images'''
    for img in images:
        img.set_outfile()
        cv2.imwrite(img.transformed,img.outfile)

class ImageData():
    '''bucket in which to hold an image and associated metadata. 
    This is mostly for I/O, computational steps will be done elsewhere'''
    def __init__(self, args, fname, read_mode=cv2.IMREAD_GRAYSCALE):
        self.args = args
        self.fname = fname
        basename,ext = os.path.splitext(os.path.basename(fname))
        self.basename = basename
        self.in_ext = ext
        self.image = cv2.imread(fname,read_mode)
        self.transformed = None # this will hold e.g. dots or regions
        self.outfile = None

    def set_outfile(self):
        '''default is to use basename with edit to ensure no overwrite '''
        out_format = self.in_ext
        if self.args.out_format:
            out_format = self.args.out_format
        self.outfile = os.path.join(args.output,"{}_cv2.{}".format(self.basename,out_format))

    def imwrite(self):
        '''a wrapper for cv2.imwrite'''
        cv2.imwrite(self.outfile,self.transformed)

class CellCountImage(ImageData):
    ''' more specific bucket in which to hold an image and metadata for an 
    image which cell counting will be performed on.

    >>> testArgs = parse_args(['-o', '/fake/out/', '-i', 'fake/in.png'])
    >>> cci = CellCountImage(testArgs,'fake/in.png')
    >>> cci.count 
    'uncounted'
    >>> cci.outfile 

    >>> cci.set_outfile()
    >>> cci.outfile 
    '/fake/out/in.uncounted.png'
    '''
    def __init__(self, args, fname, read_mode=cv2.IMREAD_GRAYSCALE):
        self.count = 'uncounted' # sentinel value, to be changed to an integer
        super().__init__(args, fname, read_mode=cv2.IMREAD_GRAYSCALE)

    def set_outfile(self):
        ''' add cell number to format'''
        out_format = self.in_ext.lstrip('.')
        if self.args.out_format:
            out_format = self.args.out_format
        outfile_fname = "{}.{}.{}".format(self.basename,self.count,out_format)
        self.outfile = os.path.join(self.args.output,outfile_fname)

def parse_args(in_args):
    '''separate argparse function to make construcion of test objects much easier'''
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True,
           help='path to the output base')
    ap.add_argument('-i', '--input', required=True,nargs='+' ,
           help='input files, separated by spaces')
    ap.add_argument('-f', '--out-format',
           help='path to the output base. If not set, defaults to input format.')
    args = ap.parse_args(in_args)
    args.allowed_input_ext = ['png']
    return args

def validate_in_images(args):
    ''' takes in path to images, validates that they exist and are of the proper extension
    >>> testArgs = parse_args(['-o', '/fake/out/', '-i', 'fake/in.png', './test_images/124cell.png'])
    >>> try: 
    ...     validate_in_images(testArgs)
    ... except OSError:
    ...     pass
    ... else:
    ...     raise AssertionError('Should have raised an exception')
    fake/in.png
    '''
    errlist = []
    for im in args.input:
        basename,ext = os.path.splitext(im)
        if ext.lower().lstrip('.') not in args.allowed_input_ext:
            errlist.append(im)
        elif not os.path.isfile(im):
            errlist.append(im) 
    if errlist:
        for im in errlist:
            print(im)
        raise OSError('Input files listed above not found')

def ensure_outpath(args):
    '''creates outpath if needed'''
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
def blob_method():
    #args = parse_args(sys.argv[1:])
    args = parse_args(['-o', './test_out/', '-i', './test_images/124cell.png', './test_images/160cell.png', './test_images/176cell.png'])
    ensure_outpath(args)
    validate_in_images(args)
    images = []
    for image_name in args.input:
        images.append(CellCountImage(args,image_name))

    detector = SimpleBlobCellDetector()
    
    for image in images:
        detector.simple_blob_cell_detection(image)
        image.set_outfile()
        image.imwrite()

def area_method():
    cell_agnostic_counter = sklearn.pipeline.Pipeline(steps=[
        ('get_cell_area', ThresholdCellArea()),
        ('linear_model', sklearn.linear_model.LinearRegression()),
        ])

def test():
    import doctest
    count, _ = doctest.testmod()
    if count == 0:
        print('Docs passed!')
    
if __name__ == '__main__':
    blob_method()
