from __future__ import print_function
import numpy as np
import os
import scipy
import time

from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, felzenszwalb, slic
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


suffix = ""
ndvi = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_ndvi_subset.tif"
RASTER_DATA_FILE = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_4band_subset"+suffix+".tif"
TRAIN_DATA_PATH = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/SubTrain"
TEST_DATA_PATH = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/SubTest"

print ("input paths set")

driverTiff = gdal.GetDriverByName("GTiff")
raster_dataset = gdal.Open(RASTER_DATA_FILE, gdal.GA_ReadOnly)
ndvi_ds = gdal.Open(ndvi, gdal.GA_ReadOnly)
ndvidat = ndvi_ds.GetRasterBand(1).ReadAsArray()
print ("raster opened")
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()
n_bands = raster_dataset.RasterCount
bands_data = []
print ("n bands = "+str(n_bands))
for b in range(1, n_bands+1):
    band = raster_dataset.GetRasterBand(b)
    dat = band.ReadAsArray()
    #dat = np.ma.masked_where(dat == 0,dat)
    bands_data.append(dat)

bands_data = np.dstack(b for b in bands_data)
#bands_data = np.dstack((bands_data,ndvidat))
img = exposure.rescale_intensity(bands_data)

print ("data stacked "+str(img.shape))

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds

def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels

def segment_features(segment_pixels):
    """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
    features = []
    n_pixels, n_bands = segment_pixels.shape
    for b in range(n_bands):
        stats = scipy.stats.describe(segment_pixels[:,b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if n_pixels == 1:
            # scipy.stats.describe raises a Warning and sets variance to nan
            band_stats[3] = 0.0  # Replace nan with something (zero)
        features += band_stats
    return features

print ("helper functions defined")

print ("creating multichannel felz segments")
start = time.time()
segments_felz = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
end = time.time()
print("felzmulti segments done " + str(end-start))

felzseg = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/segments_quick_subset"+suffix+".tif",
                            raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
felzseg.SetGeoTransform(geo_transform)
felzseg.SetProjection(proj)
felzband = felzseg.GetRasterBand(1)
felzband.WriteArray(segments_felz)
felzband.FlushCache()
print ("segments written to raster")
segments = segments_felz
segment_ids = np.unique(segments)


rows, cols, n_bands = img.shape
files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
classes_labels = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(TRAIN_DATA_PATH, f) for f in files if f.endswith('.shp')]
print(shapefiles)

ground_truth = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

classes = np.unique(ground_truth)[1:]  # 0 doesn't count
len(classes)

segments_per_klass = {}
for klass in classes:
    segments_of_klass = segments[ground_truth==klass]
    segments_per_klass[klass] = set(segments_of_klass)
    print("Training segments for class %i: %i" % (klass, len(segments_per_klass[klass])))

accum = set()
intersection = set()
for class_segments in segments_per_klass.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "intersection length is not zero "+str(len(intersection))

train_img = np.copy(segments)
threshold = train_img.max() + 1
for klass in classes:
    klass_label = threshold + klass
    for segment_id in segments_per_klass[klass]:
        train_img[train_img == segment_id] = klass_label
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

print ("computing object vectors. This is the longest process.")
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    objects = []
    objects_ids = []
    for segment_label in segment_ids:
        segment_pixels = img[segments == segment_label]
        segment_model = segment_features(segment_pixels)
        objects.append(segment_model)
        # Keep a reference to the segment label
        objects_ids.append(segment_label)

    print("Created %i objects" % len(objects))

print ("object vectors computed")

training_labels = []
training_objects = []
for klass in classes:
    class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
    training_labels += [klass] * len(class_train_objects)
    print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
    training_objects += class_train_objects
print ("training classifier")
classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print ("classifying segments")
predicted = classifier.predict(objects)
print ("propogating classification to pixels")
clf = np.copy(segments)

for segment_id, klass in zip(objects_ids, predicted):
    clf[clf==segment_id] = klass

print("classifications propogated to pixels")

mask = np.sum(img, axis=2)
mask[mask>0.0]=1.0
mask[mask==0.0]=-1.0
clf = np.multiply(mask, clf)
clf[clf<0.0]=-9999.0
clfds = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/class_quick"+suffix+".tif",
                             raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
clfds.SetGeoTransform(geo_transform)
clfds.SetProjection(proj)
clfband = clfds.GetRasterBand(1)
clfband.SetNoDataValue(-9999.0)
clfband.WriteArray(clf)
clfband.FlushCache()
print ("classified segments written to raster")

shapefiles = [os.path.join(TEST_DATA_PATH, "%s.shp"%c) for c in classes_labels]
verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
for_verification = np.nonzero(verification_pixels)

verification_labels = verification_pixels[for_verification]
predicted_labels = clf[for_verification]

cm = metrics.confusion_matrix(verification_labels, predicted_labels)

def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()

print_cm(cm, classes_labels)

print("Classification accuracy: %f" % metrics.accuracy_score(verification_labels, predicted_labels))

print("Classification report:\n%s" % metrics.classification_report(verification_labels, predicted_labels,target_names=classes_labels))