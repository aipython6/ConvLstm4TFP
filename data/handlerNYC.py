import pymysql
from shapely.geometry import Point
from shapely.geometry import box
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class handlerNyc():
    def __init__(self):
        pass

    def getConnection(self):
        params = {'host': 'localhost', 'user': 'root', 'passwd': '123456', 'database': 'nyctaxi'}
        db = pymysql.connect(host=params['host'], user=params['user'], password=params['passwd'], database=params['database'])
        cur = db.cursor()
        return db, cur

    def get_train(self) -> tuple:
        db, cur = self.getConnection()
        sql = '''
          select * from train_clean where time BETWEEN '2015-02-18 00:00' and '2015-03-01 23:59'
      '''
        cur.execute(sql)
        items = cur.fetchall()
        return items

    def get_test(self) -> tuple:
        db, cur = self.getConnection()
        sql = '''
          SELECT pickup_longitude, pickup_latitude,dropoff_longitude, dropoff_latitude,
          DATE_FORMAT(
          concat( date( pickup_datetime ), ' ', HOUR ( pickup_datetime ), ':', floor( MINUTE ( pickup_datetime ) / 30 ) * 30 ),
          '%Y-%m-%d %H:%i'
          ) AS time 
          FROM test order by time
          '''
        cur.execute(sql)
        items = cur.fetchall()
        return items

    # total 200 regions
    def gon_instances(self) -> list:
        # (74.07173549999993-73.75255799999995)*111/20 = 1.771435125km;(40.797089400000004-40.568973)*111/10 = 2.53209204km
        min_longitude, max_longitude, min_latitude, max_latitude = math.fabs(-73.75255799999995), math.fabs(-74.07173549999993), 40.568973, 40.797089400000004
        gon_instances = list()
        # for longitude, grid = 0.01595887499999904
        avg_longitude_20 = (max_longitude - min_longitude) / 20
        # for latitude, grid = 0.022811640000000466
        avg_latitude_10 = (max_latitude - min_latitude) / 10
        index = 0
        lo_x, la_y = min_longitude, min_latitude
        # seg NYC's map to 10 * 20 regions
        for _ in range(1, 21):
            lo_xt = lo_x + avg_longitude_20
            for _ in range(1, 11):
                index += 1
                t = dict()
                la_yt = la_y + avg_latitude_10
                # create box
                polygon = box(lo_x, la_y, lo_xt, la_yt)
                t['index'] = index
                t['gon'] = polygon
                # for train list's shape(1149,), for test list's shape(288,)
                t['in'] = list()
                t['out'] = list()
                gon_instances.append(t)
                la_y += avg_latitude_10
            # one longitude add avg_longitude_20 in i-th
            lo_x += avg_longitude_20
            # when i-th latitude end, set la_y = min_latitude
            la_y = min_latitude
        return gon_instances

    # Generate 200_in dict and 200_out dict
    def init_in_out(self):
        in_200 = list()
        out_200 = list()
        for index in range(1, 201):
            in_200.append({'index': index, 'in': 0})
            out_200.append({'index': index, 'out': 0})
        return in_200, out_200

    def test(self):
        instances = self.gon_instances()
        for _, i in enumerate(self.get_test()):
            for _, item in enumerate(instances):
                if item['gon'].contains(Point(math.fabs(float(i[0])), math.fabs(float(i[1])))):
                    item['out'] += 1
                if item['gon'].contains(Point(math.fabs(float(i[2])), math.fabs(float(i[3])))):
                    item['in'] += 1
        return instances

    # fill regions
    def fill_gon(self) -> list:
        instances = self.gon_instances()
        t_in, t_out = self.init_in_out()
        t_time = '2015-02-18 00:00'     # init t_time
        for index, data in enumerate(self.get_train()):
            for _, item in enumerate(instances):
                if item['gon'].contains(Point(math.fabs(float(data[0])), math.fabs(float(data[1])))):
                    # find target item's index
                    target_out = list(filter(lambda c: c['index'] == item['index'], t_out))
                    target_out[0]['out'] += 1
                if item['gon'].contains(Point(math.fabs(float(data[2])), math.fabs(float(data[3])))):
                    target_in = list(filter(lambda c: c['index'] == item['index'], t_in))
                    target_in[0]['in'] += 1
            # time change, append t-th data to corresponding in and out
            if t_time != data[4]:
                t_time = data[4]
                for _, in_ in enumerate(t_in):
                    for _, item in enumerate(instances):
                        if in_['index'] == item['index']:
                            item['in'].append(in_['in'])
                for _, out_ in enumerate(t_out):
                    for _, item in enumerate(instances):
                        if out_['index'] == item['index']:
                            item['out'].append(out_['out'])
                # init t_in and t_out when i-th finished
                t_in, t_out = self.init_in_out()
            print('{} finished...'.format(index+1))
            print('--------------')
        return instances

    # generate numpy array
    def gen_np_array(self, instances: list, time_step: int) -> np.array:
        '''
        @params
        instances: gon instance
        time_step: 24/0.5*48(48 time-step one day, total 48 days)
        '''
        time_interval = list()
        for index in range(0, time_step):
            region = list()
            for _, ins in enumerate(instances):
                region.append([ins['in'][index], ins['out'][index]])
            time_interval.append(region)
        result = np.array(time_interval).reshape(time_step, 10, 20, 2).astype('float64')  # region: 10 * 20, in/out:2
        return result   # 4D array, shape(time_step,10,20,2)

    def plot(self, array):
        sns.heatmap(array, cmap='rainbow')
        plt.savefig('./train1.jpg', dpi=400, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    start = datetime.now()

    h = handlerNyc()
    ins = h.fill_gon()
    # for i in ins:
    #     sum1.append(sum(i['out']))
    #     sum2.append(sum(i['in']))
    # print((sum1))   # out: 9670
    # print((sum2))   # in: 9430
    time_step = 0
    for i in ins:
        time_step = len(i['in'])
    np.save('./test.npz', h.gen_np_array(instances=ins, time_step=time_step))

    end = datetime.now()
    print('total {}s'.format((end-start).seconds))
