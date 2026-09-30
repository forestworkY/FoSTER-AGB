import ee

LANDSAT_SCALE = 30
SELECT_WINDOW_PIXELS = 30
FINAL_WINDOW_PIXELS = 33
QUALIFIED_VALID_FRAC = 0.8
MAX_IMAGES_USED = 5
RAW_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
INDEX_BANDS = ["NDVI", "NDMI", "NBR"]
STAT_BANDS = RAW_BANDS + INDEX_BANDS


def make_square_window_from_point(point_geom, pixel_size, scale=30):
    side_m = ee.Number(pixel_size).multiply(scale)
    half_m = side_m.divide(2)
    return point_geom.buffer(half_m).bounds()


def prep_landsat_l2(img):
    optical = img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']).multiply(2.75e-05).add(-0.2)
    qa_pixel = img.select('QA_PIXEL')
    qa_radsat = img.select('QA_RADSAT')
    qa_dilated_cloud = qa_pixel.bitwiseAnd(1 << 1).gt(0).rename('qa_dilated_cloud')
    qa_cirrus = qa_pixel.bitwiseAnd(1 << 2).gt(0).rename('qa_cirrus')
    qa_cloud = qa_pixel.bitwiseAnd(1 << 3).gt(0).rename('qa_cloud')
    qa_shadow = qa_pixel.bitwiseAnd(1 << 4).gt(0).rename('qa_shadow')
    qa_snow = qa_pixel.bitwiseAnd(1 << 5).gt(0).rename('qa_snow')
    qa_saturated = qa_radsat.neq(0).rename('qa_saturated')
    qa_clear = qa_dilated_cloud.Not().And(qa_cirrus.Not()).And(qa_cloud.Not()).And(qa_shadow.Not()).And(qa_snow.Not()).And(qa_saturated.Not()).rename('qa_clear')
    mask = qa_clear
    ndvi = optical.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = optical.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')
    nbr = optical.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
    valid = ee.Image.constant(1).updateMask(mask).unmask(0).rename('valid')
    out = optical.addBands([ndvi, ndmi, nbr]).updateMask(mask).addBands(valid).addBands([qa_cloud.unmask(0), qa_shadow.unmask(0), qa_snow.unmask(0), qa_cirrus.unmask(0), qa_saturated.unmask(0), qa_clear.unmask(0)])
    return out.copyProperties(img, img.propertyNames())


def get_landsat_collection(start_date, end_date, geom):
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate(start_date, end_date).filterBounds(geom).map(prep_landsat_l2)
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(start_date, end_date).filterBounds(geom).map(prep_landsat_l2)
    return l8.merge(l9)


def add_scene_quality_props(img, window_geom):
    stats = img.select(['valid', 'qa_cloud', 'qa_shadow', 'qa_snow', 'qa_cirrus', 'qa_saturated', 'qa_clear']).reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True), geometry=window_geom, scale=LANDSAT_SCALE, maxPixels=100000000.0)
    valid_frac = ee.Number(stats.get('valid_mean'))
    valid_pixel_n = ee.Number(stats.get('valid_sum'))
    scene_id = ee.String(ee.Algorithms.If(img.propertyNames().contains('LANDSAT_PRODUCT_ID'), img.get('LANDSAT_PRODUCT_ID'), img.get('system:index')))
    scene_date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
    cloud_cover = ee.Number(ee.Algorithms.If(img.propertyNames().contains('CLOUD_COVER'), img.get('CLOUD_COVER'), -9999))
    return img.set({'scene_id': scene_id, 'scene_date': scene_date, 'valid_frac': valid_frac, 'valid_pixel_n': valid_pixel_n, 'cloud_cover_scene': cloud_cover, 'cloud_ratio': ee.Number(stats.get('qa_cloud_mean')), 'shadow_ratio': ee.Number(stats.get('qa_shadow_mean')), 'snow_ratio': ee.Number(stats.get('qa_snow_mean')), 'cirrus_ratio': ee.Number(stats.get('qa_cirrus_mean')), 'saturated_ratio': ee.Number(stats.get('qa_saturated_mean')), 'clear_ratio': ee.Number(stats.get('qa_clear_mean'))})


def screen_scenes_in_window(start_date, end_date, point_geom):
    select_window = make_square_window_from_point(point_geom, pixel_size=SELECT_WINDOW_PIXELS, scale=LANDSAT_SCALE)
    col = get_landsat_collection(start_date, end_date, point_geom).map(lambda img: add_scene_quality_props(img, select_window))
    candidate_col = col.filter(ee.Filter.notNull(['valid_frac']))
    qualified_col = candidate_col.filter(ee.Filter.gte('valid_frac', QUALIFIED_VALID_FRAC))
    n_candidate = candidate_col.size()
    n_qualified = qualified_col.size()
    used_col = ee.ImageCollection(ee.Algorithms.If(n_qualified.lte(MAX_IMAGES_USED), qualified_col, qualified_col.sort('valid_frac', False).limit(MAX_IMAGES_USED)))
    return {'candidate_col': candidate_col, 'qualified_col': qualified_col, 'used_col': used_col, 'n_candidate': n_candidate, 'n_qualified': n_qualified, 'n_used': used_col.size()}


def safe_get_number(stats_dict, key, default_value=-9999):
    value = stats_dict.get(key)
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), default_value, value))


def attach_final_w33_stats(feature):
    year = ee.Number(feature.get('INVYR')).int()
    point_geom = feature.geometry()
    w1_start = ee.Date.fromYMD(year, 7, 1)
    w1_end = ee.Date.fromYMD(year, 9, 1)
    res1 = screen_scenes_in_window(w1_start, w1_end, point_geom)
    w2_start = ee.Date.fromYMD(year, 6, 15)
    w2_end = ee.Date.fromYMD(year, 9, 16)
    res2 = screen_scenes_in_window(w2_start, w2_end, point_geom)
    w3_start = ee.Date.fromYMD(year, 6, 1)
    w3_end = ee.Date.fromYMD(year, 10, 1)
    res3 = screen_scenes_in_window(w3_start, w3_end, point_geom)
    use_w1 = res1['n_qualified'].gte(1)
    use_w2 = res2['n_qualified'].gte(1)
    use_w3 = res3['n_qualified'].gte(1)
    selected_used_col = ee.ImageCollection(ee.Algorithms.If(use_w1, res1['used_col'], ee.Algorithms.If(use_w2, res2['used_col'], ee.Algorithms.If(use_w3, res3['used_col'], ee.ImageCollection([])))))
    time_window_used = ee.String(ee.Algorithms.If(use_w1, '07-01_08-31', ee.Algorithms.If(use_w2, '06-15_09-15', ee.Algorithms.If(use_w3, '06-01_09-30', 'none'))))
    n_candidate_images = ee.Number(ee.Algorithms.If(use_w1, res1['n_candidate'], ee.Algorithms.If(use_w2, res2['n_candidate'], ee.Algorithms.If(use_w3, res3['n_candidate'], 0))))
    n_qualified_images = ee.Number(ee.Algorithms.If(use_w1, res1['n_qualified'], ee.Algorithms.If(use_w2, res2['n_qualified'], ee.Algorithms.If(use_w3, res3['n_qualified'], 0))))
    n_images_used = ee.Number(ee.Algorithms.If(use_w1, res1['n_used'], ee.Algorithms.If(use_w2, res2['n_used'], ee.Algorithms.If(use_w3, res3['n_used'], 0))))
    feature = feature.set({'final_window_pixels': FINAL_WINDOW_PIXELS, 'select_window_pixels': SELECT_WINDOW_PIXELS, 'qualified_valid_frac_threshold': QUALIFIED_VALID_FRAC, 'max_images_used': MAX_IMAGES_USED, 'time_window_used': time_window_used, 'n_candidate_images': n_candidate_images, 'n_qualified_images': n_qualified_images, 'n_images_used': n_images_used})

    def set_no_data(feat):
        updates = {'has_valid_landsat': 0}
        updates['w33_valid_frac'] = -9999
        updates['w33_valid_pixel_n'] = -9999
        updates['w33_obs_count_mean'] = -9999
        updates['w33_obs_count_min'] = -9999
        updates['w33_obs_count_max'] = -9999
        updates['w33_cloud_ratio_mean'] = -9999
        updates['w33_shadow_ratio_mean'] = -9999
        updates['w33_snow_ratio_mean'] = -9999
        updates['w33_cirrus_ratio_mean'] = -9999
        updates['w33_saturated_ratio_mean'] = -9999
        updates['w33_clear_ratio_mean'] = -9999
        updates['w33_cloud_ratio_max'] = -9999
        updates['w33_shadow_ratio_max'] = -9999
        updates['scene_cloud_cover_mean'] = -9999
        updates['scene_cloud_cover_min'] = -9999
        updates['scene_cloud_cover_max'] = -9999
        for band in STAT_BANDS:
            updates[f'w33_{band}_mean'] = -9999
        return feat.set(updates)

    def set_with_data(feat):
        composite_img = selected_used_col.median().select(STAT_BANDS)
        obs_count_img = selected_used_col.select('valid').sum().rename('obs_count')
        window_geom = make_square_window_from_point(point_geom, FINAL_WINDOW_PIXELS, LANDSAT_SCALE)
        band_stats = composite_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=window_geom, scale=LANDSAT_SCALE, maxPixels=100000000.0)
        valid_presence = obs_count_img.gt(0).rename('has_obs')
        valid_stats = valid_presence.reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True), geometry=window_geom, scale=LANDSAT_SCALE, maxPixels=100000000.0)
        obs_stats = obs_count_img.reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True), geometry=window_geom, scale=LANDSAT_SCALE, maxPixels=100000000.0)
        updates = {'has_valid_landsat': 1, 'w33_valid_frac': safe_get_number(valid_stats, 'has_obs_mean'), 'w33_valid_pixel_n': safe_get_number(valid_stats, 'has_obs_sum'), 'w33_obs_count_mean': safe_get_number(obs_stats, 'obs_count_mean'), 'w33_obs_count_min': safe_get_number(obs_stats, 'obs_count_min'), 'w33_obs_count_max': safe_get_number(obs_stats, 'obs_count_max')}
        updates['w33_cloud_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('cloud_ratio'))
        updates['w33_shadow_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('shadow_ratio'))
        updates['w33_snow_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('snow_ratio'))
        updates['w33_cirrus_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('cirrus_ratio'))
        updates['w33_saturated_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('saturated_ratio'))
        updates['w33_clear_ratio_mean'] = ee.Number(selected_used_col.aggregate_mean('clear_ratio'))
        updates['w33_cloud_ratio_max'] = ee.Number(selected_used_col.aggregate_max('cloud_ratio'))
        updates['w33_shadow_ratio_max'] = ee.Number(selected_used_col.aggregate_max('shadow_ratio'))
        updates['scene_cloud_cover_mean'] = ee.Number(selected_used_col.aggregate_mean('cloud_cover_scene'))
        updates['scene_cloud_cover_min'] = ee.Number(selected_used_col.aggregate_min('cloud_cover_scene'))
        updates['scene_cloud_cover_max'] = ee.Number(selected_used_col.aggregate_max('cloud_cover_scene'))
        for band in STAT_BANDS:
            updates[f'w33_{band}_mean'] = safe_get_number(band_stats, band)
        return feat.set(updates)
    return ee.Feature(ee.Algorithms.If(n_images_used.gt(0), set_with_data(feature), set_no_data(feature)))
