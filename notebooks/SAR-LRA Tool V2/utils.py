# define some useful functions

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam

fil_size1 = 3
fil_size2 = 3
fil_size3 = 3

def CNN(lr, loss, filtersFirstLayer, drop, input_size=(64, 64, 4)):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(filtersFirstLayer, fil_size1, padding='same', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(filtersFirstLayer, fil_size2, padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(filtersFirstLayer, fil_size3, padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)

    target_shape = (conv3.shape[1], conv3.shape[2])

    resized_tensor_3 = tf.image.resize(conv3, target_shape)

    concatenated_tensor = tf.concat([resized_tensor_3, resized_tensor_3, resized_tensor_3], axis=-1)

    pool3 = MaxPooling2D()(concatenated_tensor)

    drop1 = Dropout(drop)(pool3)
    flat = Flatten()(drop1)
    en = Dense(filtersFirstLayer * 8, activation='relu')(flat)
    out = Dense(1, activation='sigmoid')(en)
    model = Model(inputs, out)

    model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics='accuracy')
    return model

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Compute binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

    # Compute the predicted probabilities for the true class
    p_t = tf.math.exp(-bce)

    # Compute the focal loss
    focal_loss = alpha * (1 - p_t) ** gamma * bce

    return focal_loss

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


import os, time, requests, numpy as np, cv2, rasterio
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import re

# === Core runner ===============================================================

def run_lra_inference_per_rel(
    place,
    orbit,
    base_dir="deploy/VV_VH/60_12",
    weights_url=None,
    weights_local_path=None,
    size=64,
    channels=4,
    prob_thresh=0.6,
    nms_overlap=0.1,
    batch_size=512,
    filtersFirstLayer=64,
    drop=0.7,

):
    """
    Run CNN inference for a given ORBIT over a list of RELATIVE ORBIT tracks (RELs).
    Expects per-REL composites at:
        {base_dir}/{orbit}/REL_{rel}/SAR_{orbit}_REL{rel}.tif
    Writes outputs to:
        predictions/{place}_{orbit}_REL{rel}.tif  and .shp

    Either `weights_url` (downloaded on the fly) or `weights_local_path` must be provided.
    """

    assert orbit in ("DESCENDING", "ASCENDING"), "orbit must be 'DESCENDING' or 'ASCENDING'"
    if not (weights_url or weights_local_path):
        raise ValueError("Provide either weights_url or weights_local_path")

    os.makedirs("predictions", exist_ok=True)

    # --- Load model + weights ---
    print("[INFO] initializing model...")
    model = CNN(
        filtersFirstLayer=filtersFirstLayer,
        drop=drop,
        lr=0.001,
        input_size=(size, size, channels),
        loss=focal_loss,
    )

    weights_path = weights_local_path or f"model_weights_{orbit.lower()}.hdf5"
    if weights_url:
        print(f"[INFO] downloading weights for {orbit} ...")
        r = requests.get(weights_url)
        r.raise_for_status()
        with open(weights_path, "wb") as f:
            f.write(r.content)
        print(f"[INFO] weights saved to {weights_path}")

    print("[INFO] loading model weights...")
    model.load_weights(weights_path)
    print("[INFO] model ready.")

    # Base directory structure: e.g. deploy/VV_VH/60_12/DESCENDING/REL_###
    orbit_dir = os.path.join(base_dir, orbit)

    # Find all subfolders named like "REL_###"
    rel_list = []
    if os.path.exists(orbit_dir):
        for entry in os.listdir(orbit_dir):
            if os.path.isdir(os.path.join(orbit_dir, entry)):
                m = re.match(r"REL_(\d+)", entry)
                if m:
                    rel_list.append(int(m.group(1)))

    rel_list = sorted(rel_list)

    print(f"[INFO] found {len(rel_list)} REL tracks for {orbit}: {rel_list}")


    # --- Process each REL ---
    for rel in rel_list:
        rel = int(rel)
        rel_dir = os.path.join(base_dir, orbit, f"REL_{rel}")
        image_path = os.path.join(rel_dir, f"SAR_{orbit}_REL{rel}.tif")

        if not os.path.exists(image_path):
            print(f"[WARN] missing composite for {orbit} REL {rel}: {image_path} — skipping.")
            continue

        print(f"\n[INFO] loading image for {orbit} REL {rel}:\n       {image_path}")
        with rasterio.open(image_path) as src:
            # HWC bands last
            tmp = np.moveaxis(src.read(), 0, 2)     # (H, W, C)
            transform = src.transform
            crs = src.crs
            profile = src.profile

        orig = tmp[:, :, :channels].astype(np.float32)

        # --- Sliding window extraction ---
        ROI_SIZE = (size, size)
        WIN_STEP = size // 2  # 50% overlap
        rois, locs = [], []
        start = time.time()
        for (x, y, roiOrig) in sliding_window(orig, WIN_STEP, ROI_SIZE):
            roi = cv2.resize(roiOrig, ROI_SIZE, interpolation=cv2.INTER_LINEAR)
            rois.append(img_to_array(roi))
            w, h = ROI_SIZE
            locs.append((x, y, x + w, y + h))
        end = time.time()
        print("[INFO] extracted {:,} patches in {:.3f}s".format(len(rois), end - start))

        if len(rois) == 0:
            print("[WARN] no patches extracted — skipping this REL.")
            continue

        rois = np.asarray(rois, dtype=np.float32)

        # --- Prediction ---
        print("[INFO] classifying ROIs...")
        pred_datagen = ImageDataGenerator()
        ds = pred_datagen.flow(rois, batch_size=batch_size, seed=42, shuffle=False)
        t0 = time.time()
        ynew = model.predict(ds, verbose=0).ravel()
        dt = time.time() - t0
        print("[INFO] classification done in {:.3f}s".format(dt))

        # --- Threshold + NMS ---
        idx = np.where(ynew > prob_thresh)[0]
        candidate_boxes = np.array([locs[i] for i in idx], dtype=np.int32)
        if candidate_boxes.size == 0:
            print("[INFO] no boxes above threshold for this REL.")
            candidate_boxes = np.zeros((0,4), dtype=np.int32)

        boxes_nms = non_max_suppression_fast(candidate_boxes, overlapThresh=nms_overlap)
        print("[INFO] kept {:,} boxes after NMS".format(len(boxes_nms)))

        # --- Rasterize detections into a mask (1 where detected) ---
        H, W = orig.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for (x0, y0, x1, y1) in boxes_nms:
            # ensure in-bounds
            x0b = max(0, min(W-1, x0))
            x1b = max(0, min(W,   x1))
            y0b = max(0, min(H-1, y0))
            y1b = max(0, min(H,   y1))
            mask[y0b:y1b, x0b:x1b] = 1

        # --- Save GeoTIFF ---
        pred_path = os.path.join("predictions", f"{place}_{orbit}_REL{rel}.tif")
        profile_out = profile.copy()
        profile_out.update(
            dtype=rasterio.float32,
            count=1,
            width=W,
            height=H,
            compress="lzw",
        )
        with rasterio.open(pred_path, "w", **profile_out) as dst:
            dst.write(mask.astype(np.float32), 1)
        print(f"[INFO] wrote raster mask: {pred_path}")

        # --- Vectorize and save shapefile ---
        print("[INFO] vectorizing detections...")
        geoms = []
        for geom, val in rio_shapes(mask, transform=transform):
            if val == 1:
                geoms.append(shape(geom))
        if len(geoms) == 0:
            print("[INFO] no polygons to write for this REL.")
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        shp_path = pred_path + ".shp"
        if len(gdf) > 0:
            gdf.to_file(shp_path)
            print(f"[INFO] wrote shapefile: {shp_path}")
        else:
            print("[INFO] shapefile skipped (empty).")

    print("\n[INFO] done with orbit:", orbit)

import ee, os, datetime
from glob import glob
import geemap, leafmap

def build_s1_rel_composites(
    geometry,
    pre_stack_end,                # "YYYY-MM-DD"
    post_stack_start,             # "YYYY-MM-DD"
    pre_days=60,
    post_days=12,
    orbits=("DESCENDING", "ASCENDING"),
    project_path="",              # root where deploy/outputs live
    list_images=True,             # print table with incidence & coverage
    fishnet_h=0.5,                # fishnet horizontal interval (deg)
    fishnet_v=0.5,                # fishnet vertical interval (deg)
    fishnet_delta=1,              # fishnet delta
    scale=10,                     # download scale
    crs_epsg="EPSG:4326",         # output CRS
    error_margin_m=1,             # ee.ErrorMargin in meters for geom ops
    initialize_ee=False           # set True if you haven't called ee.Initialize() yet
):
    """
    Builds per-REL Sentinel-1 (VV, VH, diffVV, diffVH) composites per orbit and downloads them
    as tiled GeoTIFFs, then merges per REL into:
        {project_path}/deploy/VV_VH/60_{post_days}/{ORBIT}/REL_{rel}/SAR_{ORBIT}_REL{rel}.tif

    - geometry: ee.Geometry / ee.Feature / ee.FeatureCollection (no dissolve used)
    - pre_stack_end: end date for PRE window (inclusive) as "YYYY-MM-DD"
    - post_stack_start: start date for POST window (inclusive) as "YYYY-MM-DD"
    """
    if initialize_ee:
        ee.Initialize()

    print('SENTINEL 1 SAR IMAGE PROCESSING AND ACQUISITION !')
    print('_____________________________________________________________________________________')

    # --- normalize AOI to Geometry (no dissolve to avoid margin errors)
    em = ee.ErrorMargin(error_margin_m)
    if isinstance(geometry, ee.FeatureCollection):
        aoi = geometry.geometry(em)        # or .geometry().buffer(0) if you prefer
    elif isinstance(geometry, ee.Feature):
        aoi = geometry.geometry(em)
    else:
        aoi = geometry  # assume ee.Geometry

    # --- date windows
    pre_end_py   = datetime.datetime.strptime(pre_stack_end, "%Y-%m-%d")
    pre_start_py = pre_end_py - datetime.timedelta(days=pre_days)
    post_start_py= datetime.datetime.strptime(post_stack_start, "%Y-%m-%d")
    post_end_py  = post_start_py + datetime.timedelta(days=post_days)

    print('Setting time windows...')
    print('Pre stack start:', pre_start_py)
    print('Pre stack end  :', pre_end_py)
    print('Post stack start:', post_start_py)
    print('Post stack end  :', post_end_py)

    pre_start = ee.Date(pre_start_py)
    pre_end   = ee.Date(pre_end_py)
    post_start= ee.Date(post_start_py)
    post_end  = ee.Date(post_end_py)

    # --- paths
    inputs_root = os.path.join(project_path, 'deploy', 'VV_VH', f'60_{post_days}')
    # outputs_path= os.path.join(project_path, 'outputs')
    os.makedirs(inputs_root, exist_ok=True)
    # os.makedirs(outputs_path, exist_ok=True)

    # --- optional printer
    def list_with_angles_and_coverage(ic, label):
        aoi_area = aoi.area(1).getInfo()
        count = ic.size().getInfo()
        print(f'-- {label}: {count} items --')
        if count == 0:
            return
        lst = ic.sort('system:time_start').toList(count)
        print(f'{"DATE":<12} {"PLAT":<4} {"ORB":<4} {"REL":<5} {"INC(deg)":>9} {"COV%AOI":>9}  ID')
        for i in range(count):
            img = ee.Image(lst.get(i))
            img_id = img.id().getInfo()
            date   = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            orbit  = img.get('orbitProperties_pass').getInfo()
            rel_orb= img.get('relativeOrbitNumber_start').getInfo()
            inc    = img.get('meanIncidenceAngle').getInfo()
            inc_str= f'{inc:.2f}' if isinstance(inc, (int, float)) else 'NA'
            plat   = img_id[:3]
            inter_area = img.geometry(em).intersection(aoi, em).area(1).getInfo()
            cov_pct = (inter_area / aoi_area * 100.0) if aoi_area > 0 else 0.0
            print(f'{date:<12} {plat:<4} {orbit:<4} {str(rel_orb):<5} {inc_str:>9} {cov_pct:9.2f}  {img_id}')

    # --- fishnet
    fishnet = geemap.fishnet(aoi, h_interval=fishnet_h, v_interval=fishnet_v, delta=fishnet_delta)

    for orbit in orbits:
        print('\n===============================================================================')
        print('Orbit: ', orbit)
        print('project_path: ', project_path)
        print('training_path:', inputs_root)
        # print('outputs_path :', outputs_path)
        print('_______________________________________________________________________________')

        base = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(aoi)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.eq('orbitProperties_pass', orbit))
               )

        pre_ic  = base.filterDate(pre_start,  pre_end)
        post_ic = base.filterDate(post_start, post_end)

        if list_images:
            list_with_angles_and_coverage(pre_ic,  f'PRE ({orbit})')
            list_with_angles_and_coverage(post_ic, f'POST ({orbit})')

        # REL present in PRE ∪ POST
        rels = (pre_ic.merge(post_ic)
                    .aggregate_array('relativeOrbitNumber_start')
                    .distinct()
                    .getInfo())
        rels = sorted(int(r) for r in rels) if rels else []
        print(f'REL per {orbit}:', rels)

        if not rels:
            print(f'  >> Nessun REL trovato per {orbit} nelle finestre date. Skipping.')
            continue

        for rel in rels:
            print(f'  ---- REL {rel} ----')

            pre_rel  = pre_ic.filter(ee.Filter.eq('relativeOrbitNumber_start', rel))
            post_rel = post_ic.filter(ee.Filter.eq('relativeOrbitNumber_start', rel))

            pre_count  = pre_rel.size().getInfo()
            post_count = post_rel.size().getInfo()
            print('       Pre count :', pre_count)
            print('       Post count:', post_count)

            if pre_count == 0 or post_count == 0:
                print('       Skip REL: stack incompleto (pre o post vuoto).')
                continue

            # composites: VV/VH medians and diffs
            pre_median_VV  = pre_rel.select('VV').reduce(ee.Reducer.median()).select(0).rename('preVV')
            post_median_VV = post_rel.select('VV').reduce(ee.Reducer.median()).select(0).rename('postVV')
            diff_VV        = post_median_VV.subtract(pre_median_VV).rename('diffVV')

            pre_median_VH  = pre_rel.select('VH').reduce(ee.Reducer.median()).select(0).rename('preVH')
            post_median_VH = post_rel.select('VH').reduce(ee.Reducer.median()).select(0).rename('postVH')
            diff_VH        = post_median_VH.subtract(pre_median_VH).rename('diffVH')

            DS = (post_median_VV.addBands(post_median_VH)
                  .addBands(diff_VV).addBands(diff_VH)).clip(aoi)

            # per-orbit/per-REL folders
            rel_folder   = os.path.join(inputs_root, orbit, f'REL_{rel}')
            tiles_folder = os.path.join(rel_folder, f'VV_VH_{orbit}_REL{rel}')
            os.makedirs(tiles_folder, exist_ok=True)

            # download tiles
            geemap.download_ee_image_tiles(
                DS,
                fishnet,
                tiles_folder,
                prefix=f"VV_VH_{orbit}_REL{rel}_",
                scale=scale,
                crs=ee.Projection(crs_epsg)
            )

            # merge tiles to single GeoTIFF
            merged_out = os.path.join(rel_folder, f"SAR_{orbit}_REL{rel}.tif")
            leafmap.merge_rasters(
                tiles_folder,
                output=merged_out,
                input_pattern='*.tif'
            )
            print('       Merged raster:', merged_out)

            # cleanup tiles
            for tfile in glob(os.path.join(tiles_folder, '*.tif')):
                try:
                    os.remove(tfile)
                except Exception:
                    pass
            try:
                os.rmdir(tiles_folder)
            except OSError:
                pass

    print('\nDONE !!!')
