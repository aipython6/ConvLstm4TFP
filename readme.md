# 基于ConvLstm实现道路交通流量预测
## 1.数据集
- 原始数据来源于[纽约出租车数据](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).因为涉及到数据预处理问题,最终选择的是kaggle网站上的[数据集](https://www.kaggle.com/code/chengshiangli/generate-map-tiles-from-nyc-taxi-trip-data/notebook)

## 2.数据预处理
#### 本部分对应的文件为./data/handlerNYC.py.数据预处理分为以下几个步骤(仅供参考)：

- 2.1 首先,为了便于处理,将下载得到的csv文件的数据导入到MySQL中(也删除了一些不必要的字段),最终保留了：pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude等五个重要的字段.同时,只选择了2015.1.1~2015.3.1共60天的数据(约为100万条,按乘客上下车的坐标点来计算,就相当于有200万个点),为了加快数据处理的速度,将这100万的数据又单独保存到了另一张表中,也就是train_clean表.

- 2.2 **handlerNYC.py**中的思想是：
	
	(1)首先需要确定出纽约市整个区域的4个坐标点,这里需要参考[这个网站](https://www.travelmath.com/cities/New+York)来确定纽约市区的经纬度.为了保证尽可能的准确,需要考虑到乘客上下车的位置应该是比较集中的(而经过查询得到,经纬度中的1°就相当于11km，如果直接投影，那么许多区域所拥有点的个数可能是0),所以不能简单地将这200万个点投影到这些区域上,我们可进行一些尝试,找到最终可以投影的区域.经过多次尝试,我找到的一个比较合理的区域是A(-73.75255799999995,40.568973),B(-73.75255799999995,40.797089400000004),C(-74.07173549999993,40.568973),D(-74.07173549999993,40.797089400000004).这样就得到一个整体的大区域,然后再将这个大区域划分为10×20=200个小区域;
	
	(2)接着,遍历这200万个点,使得这些点落到这200个区域中(这些点有一部分可能在这个区域外,但对整体的影响比较小,可以忽略);
	
	(3)最后是将这200个区域的点进行转换,得到shape为(time_step,10,20,2)的4D array,并保存为numpy格式的数据即可.

## 3.模型训练和测试

- 主文件是main.py,各个模型在./model下,具体参考这些文件中的相关代码即可.

- 论文: **doi:10.1109/wcsp.2017.8171119**,[site](https://sci-hub.wf/10.1109/wcsp.2017.8171119#)

- 使用方法(安装必备的pytorch/numpy/sklearn/pymysql/shapely等)，超参数的配置在configuration.py

```shell
git clone https://github.com/aipython6/ConvLstm4TFP.git
cd ConvLstm4TFP
python main.py
```