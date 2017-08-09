import numpy as np
import os
import scipy
import time

from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, felzenszwalb, slic

suffix = "_vb"
RASTER_DATA_FILE = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_4band_subset_vb.tif"
quickfile = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/quick_test.tif"
felzfile = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/felz_test.tif"
slicfile = "C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/slic_test.tif"

driverTiff = gdal.GetDriverByName("GTiff")
raster_dataset = gdal.Open(RASTER_DATA_FILE, gdal.GA_ReadOnly)
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
print ("data stacked")
img = exposure.rescale_intensity(bands_data)
rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
print("showing figures")
# plt.figure()
# plt.imshow(rgb_img)
# plt.show()

# print ("creating quick segments")
# start = time.time()
# segments_quick = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
# end = time.time()
# print ("segmenting done " + str(end-start))
# n_segments = len(np.unique(segments_quick))
# print(n_segments)

# cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
# plt.figure()
# plt.imshow(segments_quick, interpolation='none', cmap=cmap)
# plt.show()

# quickseg = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/segments_quick_subset"+suffix+".tif",
#                              raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
# quickseg.SetGeoTransform(geo_transform)
# quickseg.SetProjection(proj)
# quickband = quickseg.GetRasterBand(1)
# quickband.WriteArray(segments_quick)
# quickband.FlushCache()
# print ("quick segments written to disk")

print ("creating felz segments for each band")
start = time.time()
band_segmentation = []
for i in range(n_bands):
    band_segmentation.append(felzenszwalb(img[:, :, i], scale=1, sigma=0.25, min_size=4))
    print ("segments done for band "+str(i))
const = [b.max() + 1 for b in band_segmentation]
segmentation = band_segmentation[0]
for i, s in enumerate(band_segmentation[1:]):
    segmentation += s * np.prod(const[:i + 1])

_, labels = np.unique(segmentation, return_inverse=True)
segments_felz = labels.reshape(img.shape[:2])
end = time.time()
print("felz segments done " + str(end-start))

felzseg = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/segments_felz_subset"+suffix+".tif",
                            raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
felzseg.SetGeoTransform(geo_transform)
felzseg.SetProjection(proj)
felzband = felzseg.GetRasterBand(1)
felzband.WriteArray(segments_felz)
felzband.FlushCache()

print ("felz segments written to disk")

# print ("creating slic segments")
# start = time.time()
# segments_slic = slic(img, n_segments=8000, multichannel=True, convert2lab=False)
# end = time.time()
# print ("segmenting done " + str(end-start))
# n_segments = len(np.unique(segments_slic))
# print(n_segments)

# cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
# plt.figure()
# plt.imshow(segments_slic, interpolation='none', cmap=cmap)
# plt.show()

# slicseg = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/segments_slic_subset"+suffix+".tif",
#                              raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
# slicseg.SetGeoTransform(geo_transform)
# slicseg.SetProjection(proj)
# slicband = slicseg.GetRasterBand(1)
# slicband.WriteArray(segments_slic)
# slicband.FlushCache()
# print ("slic segments written to disk")
#
print ("creating multichannel felz segments")
start = time.time()
segments_felz = felzenszwalb(img, scale=1, sigma=0.25, min_size=4, multichannel=True)
end = time.time()
print("felzmulti segments done " + str(end-start))

felzseg = driverTiff.Create("C:/konrad/Projects/ImageSegmentation/SpawnCreek/Intermediate/segments_felzmulti_subset"+suffix+".tif",
                            raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Float32)
felzseg.SetGeoTransform(geo_transform)
felzseg.SetProjection(proj)
felzband = felzseg.GetRasterBand(1)
felzband.WriteArray(segments_felz)
felzband.FlushCache()