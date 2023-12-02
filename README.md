# ObjectLocation



## 输入数据格式：

| Packet      | 类型                   |      |
| ----------- | ---------------------- | ---- |
| Time        | Double                 |      |
| Center_x    |                        |      |
| Center_y    |                        |      |
| w           |                        |      |
| h           |                        |      |
| cls         | Int                    |      |
| Track_id    | Int                    |      |
| Pos_uav     |                        |      |
| Pos _locate | (double,double,double) |      |
| Uav_id      | Int                    |      |
| Camera_pose |                        |      |
| Globe_id    | int                    |      |

## 滤波

 程序的输入是一个一个的packet 如上面所示。

 对于每一个输入的数据，若当前数据的时间与t_start的间隔小于一个时间阈值。

这个数据放入一个time_order列表中 ，接着读入下一个数据。

若当前数据的时间与t_start的间隔大于于一个时间阈值。

 T_start=当前数据时间

 对time_order进行滤值

 Time_filter(time_order)

   Space_filter(time_order)

![001](D:\primer\Git\figure\001.png)

 ### 时间滤波

​     这一模块的输入是一个以时间排序的输入列表time_order，是上一步处理结果的输出结果，列表中packets的时间分布在一个时间间隔中。每个输入数据packet中，有一个track_id的参数，记录了每个无人机对每个对象·的追踪id，但在当前的输入数据中，这个track_id的种类是未知的。所以就对输入列表进行遍历，提取出所有的track_id并放在一个列表track_kind中。然后用Counter函数获取track_kind的种类。在这个过程中并把输入数据按照uav_id分配给三个列表中。假设无人机的总数是3，uav_id分别为 0，1，2。

​        对每个uav的列表进行遍历，对每个track_id,按照时间先后，保存最新的一个packet，并放在列表 p_s中。这样，p_s列表中的每个packet都是其uav_id和track_id下的最新数据。

![002](D:\primer\Git\figure\002.png)

 

### 空间滤波

​       这一部分的输入是上一步时间滤波的输出p_s，其中的每个packet都是其uav_id和track_id下的最新数据。这一部分我们要为其分配一个全局globe_id。具体过程如下：

​      global_id 记录当前已经分配了多少种globe_id，global_center 记录了当前已经分配的global_id的数据的中心，包括位置中心和时间中心，这个用平均值来计算。对每一个数据，计算其与每个global_center的位置中心的距离，选取最小距离对应的global_center，若这个距离小于一个阈值，且当前的packet与这个对应的global_center处于不通的uav_id下，那么这个数据就是属于这个global_center的，并更新global_center中的pos和time和globe_id。

​      若这个数据经过以上步骤处理后，并不属于global_center中的每一类，那么就给这个数据分配一个新的globa_id，并更新global_center。

![003](D:\primer\Git\figure\003.png)

