# coding: utf-8

# Usage:
#     python project8.py
#
# Options:
#   --hide, -h: does all the image analysis without showing the images


# The following code analyzes three images (circles, objects and peppers)
# and performs the following tasks over each of them:
#   - Thresholding
#   - Object count
#   - Find centers of objects

import numpy as np
import pylab
import pymorph
import mahotas
import sys
from scipy import ndimage
from collections import defaultdict


IMAGE_FILES = ['circles.png', 'objects.png', 'peppers.png']


SHOW_IMAGES = True


def show(image, figure_number=None):
    pylab.imshow(image)
    pylab.show()


def mShow(image):
    if SHOW_IMAGES:
        show(image)


def loadAndFormat(imageFile):
    try:
        image = mahotas.imread(imageFile, as_grey=True)
    except ValueError:
        # if image could not be read as grey_scale, we read as color and then
        # keep just the first channel
        image = mahotas.imread(imageFile)
        image = image[:, :, 0]

    # the thresholding algorithms require an array of integer type
    max = np.amax(image)
    min = np.amin(image)
    image = (image - min) / (max - min)
    image *= 255
    image = image.astype(np.uint8)

    return image


def getThresholdedImage(image):
    T = mahotas.thresholding.otsu(image)
    thresholdedImage = image > T
    return thresholdedImage


def freqTable(lista):
    d = defaultdict(int)
    for item in lista:
        d[item] += 1
    return d


def multiThreshold(img, nThresholds):
    # nThresholds is the number of thresholds we try to extract
    # depending on histogram grouping, they could be less
    histo = np.histogram(img)
    histo = sorted(zip(histo[1], histo[0]))
    groupSize = img.size * 1.0 / (nThresholds + 1)
    seen = 0
    thresholds = []
    for (value, freq) in histo:
        seen += freq
        if seen > (len(thresholds) + 1) * groupSize:
            thresholds.append(value)
    mthrImage = img.copy()

    # we assign to each pixel the colour of
    # the first threshold on its right
    for i in range(len(img)):
        row = img[i]
        for j in range(len(row)):
            pixel = row[j]
            t = 0
            while t < len(thresholds) and pixel > thresholds[t]:
                t += 1
            if t == len(thresholds):
                t -= 1
            mthrImage[i][j] = thresholds[t]
    return mthrImage


def filterRegions(labeled, onlyBiggerThan):
    labeled = labeled.copy()
    regionSizes = freqTable(labeled.flatten())
    nrRegions = len(regionSizes)

    # this should be a float indicating what portion
    # of the total size a region must occupy to be counted as good
    minSize = onlyBiggerThan * labeled.size
    bigRegions = [k for k in regionSizes.keys() if regionSizes[k] > minSize]

    for row in labeled:
        for j in range(len(row)):
            if not row[j] in bigRegions:
                row[j] = 0

    nrRegions = len(set(labeled.flatten())) - 1

    # regions must be relabeled since the center algorithm takes
    # only labelsets of the form range(m)

    labels = list(set(labeled.flatten()))
    renaming_dict = {labels[i]: i for i in range(len(labels))}

    for row in labeled:
        for j in range(len(row)):
            row[j] = renaming_dict[row[j]]

    mShow(labeled)
    return labeled, nrRegions


def main():
    # Load and show original images
    pylab.gray()  # set gray scale mode
    print
    print "0. Reading and formatting images..."
    images = {f: loadAndFormat(f) for f in IMAGE_FILES}
    for f in IMAGE_FILES:
        mShow(images[f])

    ###########################
    # -----> Thresholding
    print
    print "1. Thresholding images..."
    thresholdedImages = {f: getThresholdedImage(images[f]) for f in IMAGE_FILES}
    for name in IMAGE_FILES:
        mShow(thresholdedImages[name])

    ###########################
    # -----> Count objects
    # 1st attempt: label the thresholded image from task 1
    print
    print "2. Object counting"
    pylab.jet()  # back to color mode

    print "\t1st approach: Label thresholded images"
    for name in IMAGE_FILES:
        labeled, nrRegions = ndimage.label(thresholdedImages[name])
        print "\t" + name + ": " + str(nrRegions)
        mShow(labeled)

    # 2nd attempt: Changing threshold level
    print
    print "\t2nd approach: Tuned thresholds"
    # For 'objects.png' some objects are very small (e.g.: screw) or
    # have many shades (e.g.: spoon) which makes them disappear or appear
    # fragmented after thresholding.
    # The advantage of this image is that the background is very dark,
    # so we can try using a lower threshold to make all shapes more definite

    objImage = images['objects.png']
    T = mahotas.thresholding.otsu(objImage)
    thresholdedImage = objImage > T * 0.7

    # Looks better, but...
    labeled, nrRegions = ndimage.label(thresholdedImage)
    print '\tobjects.png' + ": " + str(nrRegions)
    # it returns 18!

    # 3rd attempt: Smoothing before thresholding
    print
    print "\t3rd approach: Smoothing + Tuned threshold"
    # Let's apply some Gaussian smoothing AND a lower threshold
    smoothImage = ndimage.gaussian_filter(objImage, 3)
    T = mahotas.thresholding.otsu(smoothImage)
    thresholdedImage = smoothImage > T * 0.7
    labeled, nrRegions = ndimage.label(thresholdedImage)
    print '\tobjects.png' + ": " + str(nrRegions)

    # it worked! Let's save the labeled images for later
    # (we will use them for center calculation)
    labeledImages = {}
    labeledImages['objects.png'] = (labeled, nrRegions)
    mShow(labeled)

    # Let's see if this approach works on the other images
    for name in ['circles.png', 'peppers.png']:
        img = images[name]
        smoothImage = ndimage.gaussian_filter(img, 3)
        T = mahotas.thresholding.otsu(smoothImage)
        thresholdedImage = smoothImage > T * 0.7
        labeled, nrRegions = ndimage.label(thresholdedImage)
        print '\t' + name + ": " + str(nrRegions)

    # Again no luck with the circles!
    # (We will take a closer look at the peppers later)
    # 4th attempt:
    # 'circles.png': The problem is that some circles appear "glued together".
    # Let's try another technique:
    #    - smoothing the picture with a Gaussian filter
    #    - then searching for local maxima and counting regions
    #        (smoothing avoids having many scatter maxima and a higher level
    #         must be used than in the previous attempt)
    #    - use watershed with the maxima as seeds over the thresholded image
    #       to complete the labelling of circles
    print
    print "\t4th approach: Smoothing + Local maxima + Watershed"

    smoothImage = ndimage.gaussian_filter(images['circles.png'], 10)
    localmaxImage = pymorph.regmax(smoothImage)

    # A distance transform must be applied before doing the watershed
    dist = ndimage.distance_transform_edt(thresholdedImages['circles.png'])
    dist = dist.max() - dist
    dist -= dist.min()
    dist = dist / float(dist.ptp()) * 255
    dist = dist.astype(np.uint8)

    seeds, nrRegions = ndimage.label(localmaxImage)
    labeled = pymorph.cwatershed(dist, seeds)
    print "\t" + 'circles.png' + ": " + str(nrRegions)

    # worked right only for 'circles.png' !
    labeledImages['circles.png'] = (labeled, nrRegions)
    mShow(labeled)

    print
    print "\t5th approach: Smoothing + Multi-threshold +" +\
            " Morphology labeling + Size filtering"
    # 5th approach (only peppers left!)
    imagePeppers = images['peppers.png']
    # Problems with peppers are:
    #  - very different colours, they cause thresholding to work poorly
    #  - each pepper has some brighter parts which are detected as local maxima
    # We propose to address those issues as follows:
    #  - gaussian filter to smooth regions of light or shadow within each pepper
    smoothImage = ndimage.gaussian_filter(imagePeppers, 2)

    #  - instead of thresholding to create a binary image,
    #    create multiple thresholds to separate the most frequent colors.
    #     In this case, 3 thresholds will be enough
    mthrImagePeppers = multiThreshold(smoothImage, 3)

    #  - ndimage.label didn't give good results, we try another
    #     labelling algorithm
    from skimage import morphology

    labeled = morphology.label(mthrImagePeppers)

    nrRegions = np.max(labeled) + 1
    print "\t\tTotal number of regions"
    print "\t\t\t" + 'peppers.png' + ": " + str(nrRegions)
    #	- after counting regions, filter to keep only the sufficiently big ones

    filtered, nrRegions = filterRegions(labeled, 0.05)
    print "\t\tBig enough regions"
    print "\t\t\t" + 'peppers.png' + ": " + str(nrRegions)
    labeledImages['peppers.png'] = (filtered, nrRegions)

    mShow(filtered)

    ###########################
    # -----> Find center points
    print
    print "3. Centers for objects"
    for img in IMAGE_FILES:
        labeledImage, nr_objects = labeledImages[img]
        CenterOfMass = ndimage.measurements.center_of_mass
        labels = range(1, nr_objects + 1)
        centers = CenterOfMass(labeledImage, labeledImage, labels)
        centers = [(int(round(x)), int(round(y))) for (x, y) in centers]
        print '\t' + img + ": " + str(centers)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--hide', '-h']:
        SHOW_IMAGES = False
    else:
        SHOW_IMAGES = True

    main()