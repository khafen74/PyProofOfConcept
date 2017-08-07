import gdal

rgb = gdal.Open(r'C:\konrad\GIS_Data\USA\Imagery\NAIP\ImageSegmentationExample\spawn_merge_RGB.tif')
nir = gdal.Open(r'C:\konrad\GIS_Data\USA\Imagery\NAIP\ImageSegmentationExample\spawn_merge_NIR.tif')
tmp = gdal.GetDriverByName('MEM').CreateCopy('', rgb, 0)

nir_band = nir.GetRasterBand(1).ReadAsArray()
tmp.AddBand()
print "band added"
tmp.GetRasterBand(4).WriteArray(nir_band)
print "band populated"
out = gdal.GetDriverByName('GTiff').CreateCopy(r'C:\konrad\GIS_Data\USA\Imagery\NAIP\ImageSegmentationExample\spawn_merge_4band.tif',tmp,0)
print "output dataset written"

del out