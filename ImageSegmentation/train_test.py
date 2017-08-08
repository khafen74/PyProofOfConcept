from osgeo import ogr
import os
import random

all = r'C:/konrad\Projects/ImageSegmentation/SpawnCreek/AllPoints'
train = r'C:/konrad/Projects/ImageSegmentation/SpawnCreek/TrainPoints'
test = r'C:/konrad/Projects/ImageSegmentation/SpawnCreek/TestPoints'

driverShp = ogr.GetDriverByName("ESRI Shapefile")

for root, dirs, filenames in os.walk(all):
    for fn in filenames:
        if os.path.splitext(fn)[1] == ".shp":
            all_ds = ogr.Open(all+"/"+fn, 0)
            layer = all_ds.GetLayer()
            train_ds = driverShp.CreateDataSource(train + "/" + fn)
            train_lyr = train_ds.CreateLayer(train + "/" + fn, geom_type = ogr.wkbPoint, srs = layer.GetSpatialRef())
            test_ds = driverShp.CreateDataSource(test + "/" + fn)
            test_lyr = test_ds.CreateLayer(test + "/" + fn, geom_type = ogr.wkbPoint, srs = layer.GetSpatialRef())

            for feature in layer:
                outFeat = ogr.Feature(layer.GetLayerDefn())
                outFeat.SetGeometry(feature.GetGeometryRef().Clone())
                if random.random() > 0.35:
                    train_lyr.CreateFeature(outFeat)
                    train_lyr.SyncToDisk()
                else:
                    test_lyr.CreateFeature(outFeat)
                    test_lyr.SyncToDisk()
                outFeat = None


