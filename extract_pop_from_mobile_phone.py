'''
该代码采用cudf包对乒乓和漂移数据清洗应用了GPU并行计算，相比于组里的原始代码，效率有显著提升
乒乓、漂移是指定位点因为在不同基站之间频繁切换而造成的定位误差，由于关于如何处理乒乓、漂移问题的权威文献较少，故而本代码依旧参考组里面祖传的判断方式来清洗乒乓、漂移
由于cudf包需要在LINUX环境下进行安装和运行，故而windows用户建议升级win11并使用wsl2（windows subsystem for linux2)虚拟机运行Linux
本程序所使用的环境为wsl2 ubuntu 24.04 LTS, miniconda, python 3.11
创建虚拟环境并安装cudf的代码：
conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia  \
    cudf=24.08 python=3.11 'cuda-version>=12.0,<=12.5'
（该创建虚拟环境的代码来自cudf官网：https://docs.rapids.ai/install/）
在创建虚拟环境前，最好在windows上安装最新的英伟达显卡驱动，该驱动完美支持wsl，在linux系统中无需重复安装，可以使用nvidia-smi查看显卡驱动是否被wsl下的linux识别
其余所需的包见requirements.txt
非常感谢郑强文大师兄提供使用GPU并行计算加快手机数据处理这一idea
'''


# 该程序用于合并四类手机定位数据，并去除缺失值和重复值，去除研究区范围外和9.24以外的数据，处理后的文件名为data924
# data924为去除不在研究区范围以及去除9.24以外的数据
# 保留时间相同的第一条记录：data924_k1_k25为保留开始时间相同的第一条数据，以及去除记录数少于25条的用户的数据
# 漂移数据处理：data924_cleandrift为去除漂移效应后的数据
# 乒乓_优化后：data924_cleanpp为去除乒乓效应后的数据
# data924_last为去除地铁，并重新去除记录数少于25条的用户的数据
# 出行停留时间识别：data924_remain、data924_sttime、data924_stays分别为经过SMoT模型处理后提取出来的用户出行轨迹数据、各用户出行停留时间数据和用户停留点数据
from joblib import Parallel, delayed
import concurrent.futures
import shutil,copy
import gc
import warnings
from shapely import Point
import os
import pandas as pd
import numpy as np
from numba import njit
import cudf
from numba import cuda
import geopandas as gpd
from datetime import datetime, timedelta
from multiprocessing import Pool
import multiprocessing
import time
import re
import math
from geopy.distance import geodesic, great_circle
import sys, torch
from sqlalchemy import create_engine


distance_threshold = 500  # 距离阈值（单位：米）
velocity_threshold = 27.8  # 速度阈值（单位：m/s）
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半径，单位：米

    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算经纬度的差值（弧度）
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine公式
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算两点间的距离
    distance = R * c
    if distance==0:
        distance=0.1

    return distance  # 返回两点之间的距离，单位：米


wkdst = r'mobile'
warnings.filterwarnings("ignore")



def create_point(xy):
    return Point(xy)
@njit
def drift_detection_kernel(lon, lat, st, is_drift):
    for idx in range(1, len(lon) - 1):
        
        lon1, lat1, lon2, lat2, lon3, lat3 = lon[idx-1], lat[idx-1], lon[idx], lat[idx], lon[idx+1], lat[idx+1]
        st1, st2, st3 = st[idx-1], st[idx], st[idx+1]
        # 计算时间差（以秒为单位）
        delta_time13 = (st3 - st1) / 1e9
        if delta_time13 == 0:
            delta_time13 = 30
            
        # 计算地球上的两点距离
        R = 6371000  # 地球半径，单位：米
        
        phi1, phi2, phi3 = math.radians(lat1), math.radians(lat2), math.radians(lat3)
        delta_phi12 = math.radians(lat2 - lat1)
        delta_lambda12 = math.radians(lon2 - lon1)
        delta_phi13 = math.radians(lat3 - lat1)
        delta_lambda13 = math.radians(lon3 - lon1)
        delta_phi23 = math.radians(lat3 - lat2)
        delta_lambda23 = math.radians(lon3 - lon2)

        # Haversine 公式计算距离
        a12 = math.sin(delta_phi12 / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda12 / 2) ** 2
        c12 = 2 * math.atan2(math.sqrt(a12), math.sqrt(1 - a12))
        distance12 = R * c12
        if distance12==0:
            distance12=0.1

        if distance12 >= distance_threshold:
            a13 = math.sin(delta_phi13 / 2) ** 2 + math.cos(phi1) * math.cos(phi3) * math.sin(delta_lambda13 / 2) ** 2
            c13 = 2 * math.atan2(math.sqrt(a13), math.sqrt(1 - a13))
            distance13 = R * c13
            if distance13==0:
                distance13=0.1
            ave_vilo = distance13 / delta_time13
            if ave_vilo > velocity_threshold:
                is_drift[idx] = 1  # 标记为漂移点
            else:
                a23 = math.sin(delta_phi23 / 2) ** 2 + math.cos(phi3) * math.cos(phi2) * math.sin(delta_lambda23 / 2) ** 2
                c23 = 2 * math.atan2(math.sqrt(a23), math.sqrt(1 - a23))
                distance23 = R * c23
                if distance23==0:
                    distance23=0.1
                if distance12 / distance23 > 3:
                    is_drift[idx] = 1
                else:
                    is_drift[idx] = 0
        else:
            is_drift[idx] = 0

        
@njit
def pp_detection_kernel(lon, lat, st, is_pp):
    idx = 0
    while (idx < len(lon) - 3):
        lon_lat1,  lon_lat3, lon_lat4 = (lon[idx], lat[idx]), (lon[idx+2], lat[idx+2]), (lon[idx+3], lat[idx+3])
        st1, st2, st3,st4 = st[idx], st[idx+1], st[idx+2], st[idx+3]
        if lon_lat1 == lon_lat3:
            delta_time13 = (st3 - st1)/1e9
            if delta_time13<60:
                is_pp[idx],is_pp[idx+1]=1,1
                idx += 2
            else:
                idx += 1
        elif lon_lat1 == lon_lat4:
            delta_time14 = (st[idx+3]-st[idx])/1e9
            if delta_time14<60:
                is_pp[idx], is_pp[idx+1], is_pp[idx+2]=1,1,1
                idx += 3
            else:
                idx += 1
        else:
            idx += 1
if __name__=='__main__':

  study_area = gpd.read_file(r'study_area/xiamen_fishnet_within_island.shp')   # 读取研究区范围的shp文件
  study_area = study_area.to_crs('EPSG:4326')
  long_min,lat_min,long_max,lat_max = study_area.total_bounds
  #遍历读取手机定位数据
  folders = [folder for folder in os.listdir(wkdst) if os.path.isdir(os.path.join(wkdst, folder))]
  for folder_ in folders:
        c=0
        print(f'已经将临时存储4个文件的res文件清零，当前c={c}')
        res = pd.DataFrame()
        for dirpath, dirnames, filenames in os.walk(os.path.join(wkdst,folder_)):
            for f in filenames:
                if f.endswith('.csv'):
                    date_pattern = r'\d{4}-\d{2}-\d{2}'
                    c+=1
                    # 使用 re.search() 在文件路径中查找日期
                    match = re.search(date_pattern, dirpath)

                    # 如果找到匹配项，则提取日期
                    extracted_date = match.group()
                    print(f'这是{extracted_date}第{c}个文件，路径为{os.path.join(dirpath,f)}')

                    count = 0
                    print(os.path.join(dirpath, f))
                    # input()
                    timmingdata = pd.DataFrame()
                    for chunk in pd.read_csv(os.path.join(dirpath,f), chunksize=1000000, on_bad_lines='skip'):
                        if count==0:
                            column = list(chunk.columns)
                            tmid = column.index('脱敏ID')
                            lat = column.index('纬度')
                            lon = column.index('经度')
                            st = column.index('开始时间')
                        chunk.index += count * 1000000
                        # print(chunk.head())
                        chunk=chunk.dropna(subset=chunk.columns[[tmid,lat,lon,st]])  # 若存在缺失值，则将该行删除
                        count+=1
                        if count%10==0:
                          print(f"这是第{c}个文件{f}第{count}个chunk")
                        filtered_df = chunk[
                            # (pd.to_datetime(chunk.iloc[:,st]) >= pd.to_datetime(start) )&
                            # (pd.to_datetime(chunk.iloc[:,st]) < pd.to_datetime(end)) &
                            (chunk['经度'] >= float(long_min) ) &
                            (chunk['经度'] <= float(long_max)) &
                        (chunk['纬度'] >= float(lat_min) ) &
                        (chunk['纬度'] <=float(lat_max) )]
                        filtered_df.loc[:, filtered_df.columns[st]] = pd.to_datetime(filtered_df.loc[:, filtered_df.columns[st]])
                        filtered_df = filtered_df.sort_values(by=[filtered_df.columns[st],filtered_df.columns[tmid]])
                     
                        timmingdata = pd.concat([timmingdata,filtered_df])
                        timmingdata.drop_duplicates(subset=['脱敏ID','开始时间'],inplace=True)
                     
                    
                    print(f'本文件{f}（c为{c},共新增{len(timmingdata)}数据')
                    # input()
                    timmingdata.drop(columns='结束时间', inplace=True)
                    timmingdata['开始时间'] = pd.to_datetime(timmingdata['开始时间'])
                    
                    timmingdata_cudf = cudf.DataFrame.from_pandas(timmingdata)
                    del timmingdata
                    gc.collect()
                    timmingdata_cudf['开始时间'] = cudf.to_datetime(timmingdata_cudf['开始时间'])
                    

                    # 按照‘脱敏ID’和‘开始时间’排序
                    timmingdata_cudf = timmingdata_cudf.sort_values(by=['脱敏ID', '开始时间'])
                 
                    # 初始化漂移点标志
                    timmingdata_cudf['is_drift'] = 0
                    timmingdata_cudf['开始时间'] = timmingdata_cudf['开始时间'].astype('int64')
                    timmingdata_cudf.rename(columns={'脱敏ID':'ID','经度':'lon','纬度':'lat','开始时间':'st'},inplace=True)
                    grouped = timmingdata_cudf.groupby('ID')
                    #识别漂移点
                    timmingdata_cudf = grouped.apply_grouped(
                        drift_detection_kernel,
                        incols=['lon', 'lat', 'st'],  # 输入列
                        outcols={'is_drift': int},  # 输出列
                        )
                
                    print(timmingdata_cudf['is_drift'])
                                  
                    # 过滤掉漂移点数据
                    timmingdata_cudf = timmingdata_cudf[timmingdata_cudf['is_drift']==0]
                    timmingdata_cudf = timmingdata_cudf.drop(columns=['is_drift'])  # 删除标识列
                    print(f'经过漂移清洗后还剩{len(timmingdata_cudf)}条')
                    # 初始化乒乓效应标志
                    timmingdata_cudf['is_pp']=0
                    grouped = timmingdata_cudf.groupby('ID')
                    #识别乒乓点
                    timmingdata_cudf = grouped.apply_grouped(
                        pp_detection_kernel,
                        incols = ['lon', 'lat', 'st'],
                        outcols = {'is_pp':int}
                    )
                    print(timmingdata_cudf['is_pp'])
                    # 过滤掉乒乓点数据
                    timmingdata_cudf = timmingdata_cudf[timmingdata_cudf['is_pp']==0]
                    timmingdata_cudf = timmingdata_cudf.drop(columns=['is_pp'])  # 删除标识列
                    print(f'经过乒乓清洗后还剩{len(timmingdata_cudf)}条')
                    timmingdata = timmingdata_cudf.to_pandas() 
                    del timmingdata_cudf 
                    gc.collect()
                                  
                    res = pd.concat([res,timmingdata],axis=0)
                    del timmingdata
                    gc.collect()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"当前res数据共{len(res)}条,当前时间为{formatted_time}")
       
        res['st'] = pd.to_datetime(res['st'],unit='ns')
        res.drop_duplicates(subset=['ID','st'])
        res.sort_values(by=['ID','st'])
        print(f'当前的res数据共有{len(res)}条（经过去重）')
        res['time'] = res['st'].dt.floor('15T')
        print(res['time'].unique())
        print(len(res['time'].unique()))
        #基于15min时间窗口最接近中间时刻的手机定位，统计人口热力值
        mid_interval = pd.to_timedelta('7.5T')  # 15分钟时间段的中点
        res['time_to_mid'] = np.abs(res['st'] - (res['time'] + mid_interval))
        res_mid = res.loc[res.groupby(['time', 'ID'])['time_to_mid'].idxmin()]
        print(f'当前的res数据共有{len(res_mid)}条（经过去中间时刻重）')
        res_mid.sort_values(by=['ID', 'st'])
        #使用并行化创建几何点
        geometry = Parallel(n_jobs=-1, backend='loky')(delayed(create_point)(xy) for xy in zip(res_mid['lon'], res_mid['lat']))
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time.sleep(10)
        print(f'已经完成数字转点，当前的时间为{formatted_time}')
        
        
        gdf = gpd.GeoDataFrame(res_mid, geometry=geometry, crs='EPSG:4326')
     
        del res
        gc.collect()
    
        print(f'当前时间为{datetime.now()}')

        joined_gdf = gpd.sjoin(gdf, study_area, how='inner', predicate='within')
        print(f'当前时间为{datetime.now()}')
       
        gc.collect()       
        time_polygon_counts = (
            joined_gdf.groupby(['time', 'index_right'])
            .size()
            .reset_index(name='counts')
        )

        # 使用 pivot_table 将结果转换为时间和多边形的交叉表
        pivot_table = time_polygon_counts.pivot_table(index='index_right', columns='time', values='counts', fill_value=0)

       
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f'已经完成了按格网统计，当前的时间为{formatted_time}')
        tensor_temp = []
        for col in pivot_table.columns:
            gdf_shape_copy = copy.deepcopy(study_area)
            col_esri = str(col)[5:].replace(" ","_")
            print(f'当前存储的时间开始时间为{col}')
            # 每个时间段的统计结果添加到形状文件中，列名为每个时间段
            gdf_shape_copy[col_esri] = study_area.index.map(pivot_table[col]).fillna(0)
            file_name = str(col).replace(' ',"_").replace(':',"_")+'.xlsx'
            # file_name = '_'.join(str(col).split(' ')])+'.csv'
            output_base_path = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(output_base_path,'result_24h_mid_heat_correct_study_area',folder_)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            time.sleep(3)
            # gdf_shape_copy2 = gdf_shape_copy[['FID',]]
            gdf_shape_copy.drop(columns=['geometry'], inplace=True)
            tensor = gdf_shape_copy[col_esri].values
            tensor_temp.append(tensor)
            
        tensor = torch.tensor(tensor_temp, dtype=torch.float32)
        time.sleep(3)
        torch.save(tensor, 'mobile_'+extracted_date+'.pt')
        del time_polygon_counts
        del joined_gdf
        gc.collect()
