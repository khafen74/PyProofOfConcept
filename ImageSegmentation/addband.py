import gdal

rgb = gdal.Open(r'C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_4band_subset.tif')
nir = gdal.Open(r'C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_ndvi_subset.tif')
tmp = gdal.GetDriverByName('MEM').CreateCopy('', rgb, 0)

nir_band = nir.GetRasterBand(1).ReadAsArray()
tmp.AddBand()
print "band added"
tmp.GetRasterBand(5).WriteArray(nir_band)
print "band populated"
out = gdal.GetDriverByName('GTiff').CreateCopy(r'C:/konrad/Projects/ImageSegmentation/SpawnCreek/Imagery/spawn_5band_subset.tif',tmp,0)
print "output dataset written"

del out