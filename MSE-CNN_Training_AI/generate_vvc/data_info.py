import numpy as np

YUV_NAME_LIST_FULL = [
'akiyo_cif',
'aspen_1080p',
'blue_sky_1080p25',
'bowing_cif',
'bridge_close_cif',
'bridge_far_cif',
'bus_cif',
'city_4cif',
'coastguard_cif',
'container_cif',
'controlled_burn_1080p',
'crew_4cif',
'crowd_run_1080p50',
'deadline_cif',
'dinner_1080p30',
'ducks_take_off_1080p50',
'factory_1080p30',
'female150',
'flower_cif',
'flower_garden_720x480',
'football_720x480',
'football_cif',
'foreman_cif',
'galleon_720x480',
'garden_sif',
'hall_monitor_cif',
'harbour_4cif',
'Harmonic_10AsianFusion_2_1080p_30',
'Harmonic_10AsianFusion_5_1080p_30',
'Harmonic_11skateboarding_7_1080p_30',
'Harmonic_11skateboarding_9_1080p_30',
'Harmonic_12redrockvol3_2_1080p_50',
'Harmonic_12redrockvol3_5_1080p_50',
'Harmonic_13redrockvol2_2_1080p_50',
'Harmonic_13redrockvol2_9_1080p_50',
'Harmonic_14airacrobatics_2_1080p_50',
'Harmonic_14airacrobatics_3_1080p_50',
'Harmonic_16raptors_2_1080p_50',
'Harmonic_16raptors_3_1080p_50',
'Harmonic_18ANIMALS_11_1080p_50',
'Harmonic_18ANIMALS_3_1080p_50',
'Harmonic_2Rally_1_1080p_30',
'Harmonic_2Rally_2_1080p_30',
'Harmonic_3fjords_1_1080p_30',
'Harmonic_3fjords_2_1080p_30',
'Harmonic_5costa_3_1080p_30',
'Harmonic_5costa_5_1080p_30',
'Harmonic_6hongkong_2_1080p_30',
'Harmonic_6hongkong_6_1080p_30',
'Harmonic_7_1_1080p_30',
'Harmonic_7_7_1080p_30',
'Harmonic_8americanfootball_2_1080p_30',
'Harmonic_8americanfootball_7_1080p_30',
'highway_cif',
'husky_cif',
'ice_4cif',
'intros_720x480',
'in_to_tree_1080p50',
'life_1080p30',
'LiquidAssets_anemone_1080p_30',
'LiquidAssets_blackfish_1080p_60',
'LiquidAssets_boats_1080p_30',
'LiquidAssets_diver2_1080p_30',
'mad900_cif',
'male150',
'mobcal_ter_720p50',
'mobile_calendar_720x480',
'mobile_cif',
'mother_daughter_cif',
'Netflix_Aerial_2048x1080_60fps_420',
'Netflix_BarScene_2048x1080_60fps_420',
'Netflix_Boat_2048x1080_60fps_420',
'Netflix_BoxingPractice_2048x1080_60fps_420',
'Netflix_Crosswalk_2048x1080_60fps_420',
'Netflix_Dancers_2048x1080_60fps_420',
'Netflix_DinnerScene_2048x1080_60fps_420',
'Netflix_DrivingPOV_2048x1080_60fps_420',
'Netflix_FoodMarket_2048x1080_60fps_420',
'Netflix_Narrator_2048x1080_60fps_420',
'Netflix_PierSeaside_2048x1080_60fps_420',
'Netflix_RitualDance_2048x1080_60fps_420',
'Netflix_RollerCoaster_2048x1080_60fps_420',
'Netflix_SquareAndTimelapse_2048x1080_60fps_420',
'Netflix_Tango_2048x1080_60fps_420',
'Netflix_ToddlerFountain_2048x1080_60fps_420',
'Netflix_TunnelFlag_2048x1080_60fps_420',
'Netflix_WindAndNature_2048x1080_60fps_420',
'news_cif',
'old_town_cross_1080p50',
'onedarkfinal',
'pamphlet_cif',
'paris_cif',
'parkrun_ter_720p50',
'park_joy_1080p50',
'pedestrian_area_1080p25',
'RAISE_Train_1536x1024',
'RAISE_Train_2304x1536',
'RAISE_Train_2880x1920',
'RAISE_Train_768x512',
'red_kayak_1080p',
'riverbed_1080p25',
'rush_field_cuts_1080p',
'rush_hour_1080p25',
'shields_ter_720p50',
'sign_irene_cif',
'silent_cif',
'simo',
'sintel_trailer_2k_1080p24',
'snow_mnt_1080p',
'soccer_4cif',
'speed_bag_1080p',
'station2_1080p25',
'stefan_sif',
'stockholm_ter_720p5994',
'students_cif',
'sunflower_1080p25',
'tempete_cif',
'tennis_sif',
'touchdown_pass_1080p',
'tractor_1080p25',
'training',
'tt_sif',
'videoSRC003_640x360_30',
'videoSRC004_640x360_30',
'videoSRC005_640x360_30',
'videoSRC008_640x360_30',
'videoSRC009_640x360_30',
'videoSRC010_640x360_30',
'videoSRC015_640x360_30',
'videoSRC016_640x360_30',
'videoSRC019_640x360_30',
'videoSRC023_640x360_30',
'videoSRC025_640x360_30',
'videoSRC034_640x360_30',
'videoSRC035_640x360_30',
'videoSRC037_640x360_30',
'videoSRC050_640x360_30',
'videoSRC056_640x360_30',
'videoSRC062_640x360_30',
'videoSRC065_640x360_30',
'videoSRC073_640x360_30',
'videoSRC074_640x360_30',
'videoSRC075_640x360_30',
'videoSRC078_640x360_30',
'videoSRC079_640x360_30',
'videoSRC082_640x360_30',
'videoSRC083_640x360_30',
'videoSRC085_640x360_30',
'videoSRC095_640x360_24',
'videoSRC100_640x360_24',
'videoSRC102_640x360_24',
'videoSRC104_640x360_24',
'videoSRC107_640x360_24',
'videoSRC109_640x360_24',
'videoSRC111_640x360_24',
'videoSRC113_640x360_24',
'videoSRC114_640x360_24',
'videoSRC117_640x360_24',
'videoSRC122_640x360_30',
'videoSRC125_640x360_30',
'videoSRC130_640x360_30',
'videoSRC135_640x360_30',
'videoSRC136_640x360_24',
'videoSRC138_640x360_24',
'RAISE_Valid_1536x1024',
'RAISE_Valid_2304x1536',
'RAISE_Valid_2880x1920',
'RAISE_Valid_768x512',
'videoSRC149_640x360_30',
'videoSRC155_640x360_30',
'videoSRC160_640x360_24',
'videoSRC163_640x360_24',
'videoSRC170_640x360_24',
'videoSRC176_640x360_24',
'videoSRC180_640x360_24',
'videoSRC182_640x360_24',
'videoSRC183_640x360_24',
'videoSRC188_640x360_24',
'videoSRC192_640x360_24',
'videoSRC195_640x360_24',
'videoSRC198_640x360_24',
'videoSRC200_640x360_24',
'videoSRC201_640x360_24',
'videoSRC204_640x360_24',
'videoSRC213_640x360_24',
'vtc1nw_720x480',
'washdc_720x480',
'waterfall_cif',
'west_wind_easy_1080p',
'x2',
]

YUV_WIDTH_LIST_FULL = [
352,1920,1920,352,352,352,352,704,352,352,1920,704,1920,352,1920,1920,1920,1920,352,720,720,352,352,720,352,
    352,704,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,1920,
    1920,1920,1920,1920,1920,1920,1920,352,352,704,720,1920,1920,1920,1920,1920,1920,352,1920,1280,720,352,
    352,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,352,1920,
    1920,352,352,1280,1920,1920,1536,2304,2880,768,1920,1920,1920,1920,1280,352,352,1920,1920,1920,704,1920,
    1920,352,1280,352,1920,352,352,1920,1920,1920,352,640,640,640,640,640,640,640,640,640,640,640,640,640,
    640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,
    640,640,640,
1536,2304,2880,768,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,640,720,720,352,1920,1920,
]

YUV_HEIGHT_LIST_FULL = [
288,1080,1080,288,288,288,288,576,288,288,1080,576,1080,288,1080,1080,1080,1080,288,480,480,288,288,480,240,
    288,576,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,
    1080,1080,1080,1080,1080,1080,1080,288,288,576,480,1080,1080,1080,1080,1080,1080,288,1080,720,480,288,
    288,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,1080,288,1080,
    1080,288,288,720,1080,1080,1024,1536,1920,512,1080,1080,1080,1080,720,288,288,1080,1080,1080,576,1080,
    1080,240,720,288,1080,288,240,1080,1080,1080,240,360,360,360,360,360,360,360,360,360,360,360,360,360,
    360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,
    360,360,360,360,
1024,1536,1920,512,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,480,480,288,1080,1080,
]

YUV_FRAME_LIST_FULL = [
300,300,217,300,300,300,150,600,300,300,300,600,500,300,300,500,300,150,250,300,300,260,300,300,115,300,600,
    297,244,366,168,500,500,275,344,500,500,500,500,288,490,193,461,500,500,468,284,388,260,500,484,500,392,
    300,250,480,300,500,300,300,552,258,300,300,150,500,300,300,300,600,600,300,254,300,600,600,600,600,300,
    600,600,600,600,294,600,600,600,300,500,150,300,300,500,500,250,1600,1600,1600,1600,300,250,300,250,500,
    250,300,150,240,300,600,300,250,300,600,300,250,260,150,300,250,150,112,150,150,150,150,150,150,150,150,
    150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,120,120,120,120,120,120,120,120,
    120,120,120,120,120,120,120,120,
200,200,200,200,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,300,300,260,300,150,
]

assert(len(YUV_NAME_LIST_FULL) == len(YUV_WIDTH_LIST_FULL))
assert(len(YUV_NAME_LIST_FULL) == len(YUV_HEIGHT_LIST_FULL))
assert(len(YUV_NAME_LIST_FULL) == len(YUV_FRAME_LIST_FULL))