/*1.利用当前激光帧起止时刻间的imu数据计算旋转增量，IMU里程计数据计算平移增量,进而对该帧激光每一时刻的激光点进行运动畸变校正*/
/*利用相对于激光帧起始时刻的位姿增量，变换当前激光点到起始时刻激光点的坐标系下，实现校正*/
/*同时用IMU数据的姿态角（RPY，roll、pitch、yaw）、IMU里程计数据的的位姿，对当前帧激光位姿进行粗略初始化*/
#include "utility.h"
#include "lio_sam/cloud_info.h"
  
//    结构体还使用了 Eigen 库中定义的宏 EIGEN_MAKE_ALIGNED_OPERATOR_NEW，用于对齐内存，
//提高数据访问效率。EIGEN_ALIGN16 则是将结构体的内存对齐到16字节的边界。                                                                                  //对原始的点云去畸变
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D         // x y z w四个元素
    PCL_ADD_INTENSITY;      //2个宏定义  
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW                                                     //Eigen库中定义的宏，用于对齐内存
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)
//POINT_CLOUD_REGISTER_POINT_STRUCT 宏则是将该结构体注册到 PCL 库中，以便能够被 PCL 库的点云处理算法使用。
//其中，宏的第一个参数为待注册的数据结构体类型，第二个参数为一个元组，包含了数据结构体中每个成员变量的名称和别名。

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;                                           //队列长度

class ImageProjection : public ParamServer                              //类的继承
{
private:

    std::mutex imuLock;                                                 //互斥锁类型  线程同步的时候使用
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;                                      //类型   对象
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;       //用于rviz展示？
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;  //定义了一个名为imuQueue的双端队列（deque）,C++ STL（标准模板库）中的一个容器

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;
//这段代码定义了一个名为 imuTime 的指针变量，类型为 double*。并使用 new 运算符动态地分配了一个长度为 queueLength 的双精度浮点数数组，并将数组首地址赋值给 imuTime 指针变量。
// 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];                          //动态分配内存
    double *imuRotX = new double[queueLength];                          
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;                              //变换矩阵    4*4
    //当前帧原始激光点云
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 当期帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;
    // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;
 // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExtraction进行特征提取
    lio_sam::cloud_info cloudInfo;
    double timeScanCur;                 //起始时刻
    double timeScanEnd;                 //结束时刻
    std_msgs::Header cloudHeader;       //当前帧header，包含时间戳信息

    vector<int> columnIdnCountVec;


public:
    ImageProjection():deskewFlag(0)
    {
        //订阅三个话题
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);//发布去畸变点云    用于rviz展示？
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);         //发布有效激光点云  用于特征提取
        //分配内存，重置参数？
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);                                         //控制台只输出错误信息
    }
//用到了很多指针，这个操作是为了避免成为野指针
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    /*重置参数，接收每帧lidar数据都要重置这些参数*/
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}                                                //析构函数
//坐标系旋转->加锁->存入队列(deque)
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);               //imu转到雷达坐标系（仅旋转） imuConverter在头文件里

        std::lock_guard<std::mutex> lock1(imuLock);                     //锁定数据
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
//加锁->存入队列
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);                     //锁定数据
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {   
        //缓存雷达点云
        //cachepointcloud函数将雷达信息转为pcl点云格式，并进行点云数量、格式转换、稠密格式、ring、时间戳消息等检查工作。没问题就返回true
        if (!cachePointCloud(laserCloudMsg))      //没问题就不满足这个if条件 
            return;
        //IMU不是空，imu、odom序列时间覆盖Lidar时间
        if (!deskewInfo())                                              //deskewInfo函数准备了imu与odom补偿信息--包括旋转与平移
            return;
        //前面两个if语句可以理解成为LIDAR去畸变做准备。
        projectPointCloud();                                            //将点云中的点进行去除异常点、运动补偿等操作

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud  个数
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud        取出最早的一帧数据的第一个点
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();     //删除第一帧数据，释放内存
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)   //将坐标、强度、通道、时间戳赋值
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;                           //时间单位转化：纳秒->秒？
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp 第一帧数据
        cloudHeader = currentCloudMsg.header;
        //提取队列第一个做timeScanCur，之后pop出第一个，选第二个做timeScanNext。后面与imu对比时间戳？
        timeScanCur = cloudHeader.stamp.toSec();                            //点云数据的时间戳信息转换为以秒为单位的浮点数值。
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;     //结束时间(一帧数据最后一个点)  =   当前时间  +   相对时间

        // check dense flag
        //判断点云数据是否为稠密格式（点云中的每个点都有其对应的属性值），如果不是则输出错误信息并终止程序。
        if (laserCloudIn->is_dense == false)                            
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();//节点的优雅退出，不在订阅与发布；所以他比return更狠
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)    //fields表示坐标、强度、ring等字段
            {
                if (currentCloudMsg.fields[i].name == "ring")               //点云数据包含ring
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")              //点云数据包含时间戳
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {   //保证在对 imuQueue 进行读写操作时，不会因为多个线程同时访问而导致数据竞争等问题        怎么又锁一遍？是因为这个函数要用到两个队列信息？
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 当前帧对应imu原始数据处理
        // 1、遍历当前激光帧起止时刻之间的imu数据（其实是imu数据覆盖激光），初始时刻对应imu的姿态角RPY设为当前帧的初始欧拉角init
        // 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转增量，初始时刻旋转设为0(imuRotX、Y、Z、imuTime数组)
        // 注：imu数据都已经转换到lidar系下了
        //imu去畸变参数计算----用于旋转补偿（旋转数组有很多增量信息）
        imuDeskewInfo();

        // 当前帧对应imu里程计处理
        // 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
        // 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
        //里程计去畸变参数计算---用于旋转和平移补偿（仅利用起止odom得到变换矩阵增量）
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;
        // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        // 遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();
            
            // get roll, pitch, and yaw estimation for this scan
            // 提取imu姿态角RPY，作为当前lidar帧初始姿态角,这一帧数据是距离当前lidar帧最近的一帧                                                 
            if (currentImuTime <= timeScanCur)
                //载体IMU欧拉角表示转化为ROS中的欧拉角表示（来自磁力计）
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            // 超过当前激光帧结束时刻0.01s，结束
            //言外之意就是使用lidar帧重合的imu数据。比如lidar是100ms~200ms，imu的时间戳可能是99ms~201ms
            if (currentImuTime > timeScanEnd + 0.01)
                break;
            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){                                                                  //Q:这里为什么第一个角度要设为0°？
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            //这里针对的是2~其他帧imu数据
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            //这里是计算2~其他帧imu数据相对于第一帧的增量信息
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        // 从imu里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }
        
        if (odomQueue.empty())
            return;
        // 要求必须有当前激光帧时刻之前的里程计数据
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;
        //找到第一个时间大于Lidar初始时刻的odom
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }
        // 提取start0domMsg的imu里程计姿态角---四元数->欧拉角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
         // 用当前激光帧起始时刻的imu里程计，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;
        // 提取当前激光帧结束时刻的imu里程计
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 如果起止时刻对应imu里程计的协方差第一个元素不等，返回    这个协方差表明odom退化了，置信度不高
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        //变换到第一个点云帧坐标系下
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        //变换矩阵transEnd表示终止里程计消息中的位置和姿态变换。                                                                                                                                    
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 起止时刻imu里程计的相对变换              对应数学关系？
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 给定的转换中，提取XYZ以及欧拉角,通过tranBt 获得增量值(相对于起始时刻)  后续去畸变用到
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }
    //在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        //要么imuPointerFront计数大于了imu一组的数量imuPointerCur
	    //要么该次imu时间戳大于了该点时间戳
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])  //保证当前激光点的时间小于imu最后那个点的时间
                break;
            ++imuPointerFront;
        }
        //如果计数为0或该次imu时间戳小于了该点时间戳
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            //算一下该点时间戳在本组imu中的位置？
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            //按前后百分比赋予旋转量
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
     //返回值类型是pcl::PointXYZI
     /*利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿*/
    PointType deskewPoint(PointType *point, double relTime)
    {
        //这个来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        //当前点的时间 = scan时间（初始时刻）加relTime（后文的laserCloudIn->points[i].time）
        double pointTime = timeScanCur + relTime;
        //根据时间戳插值获取imu计算的旋转量与位置量（imu计算的相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        //这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用cloudHandler都会启动
	    //也就是求扫描第一个点起始姿态到世界坐标系下变换矩阵(4*4)
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        //扫描当前点从世界坐标系到目标坐标系
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        //扫描该点相对扫描本次scan第一个点的lidar变换矩阵=第一个点时lidar世界坐标系下变换矩阵的逆×当前点时lidar世界坐标系下变换矩阵
        //这里是得到当前点变换到第一个点所在坐标系的变换矩阵
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        //根据lidar位姿变换，修正点云位置
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();                                                         
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            //保留原始点云的数据XYZI
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            //点的距离太大或者太小都不要
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
            //线数范围外也不要
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            //确保rowIdn是一个整数
            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                //水平角
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                //每线一圈Horizon_SCAN（1800）的点，也就是每0.2°就有一个点
                static float ang_res_x = 360.0/float(Horizon_SCAN);
        
            //horizonAngle 为[-180,180],horizonAngle -90 为[-270,90],-round 为[-90,270], /ang_res_x 为[-450,1350]
            //+Horizon_SCAN/2为[450,2250]
            // 即把horizonAngle从[-180,180]映射到[450,2250]
            // 对应关系
            // 450    1350    1800    2250
            // -180°   0°      90°     180°
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                //将columnIdn的范围转换为[0~1800]
                //0     450     1350        1800
                //90°   +-180°   0°          90°
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            //如果线数不正确
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            //FLT_MAX 是 float 类型的最大值          满足这个条件的点是大多数点，不满足的一般为无效点       也就是说大多数情况下后边的语句没用了？
            //另一个理解角度：这个位置没有值
            //这个点有填充跳过
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            //去畸变    坐标系转换      即转换到当前帧雷达数据第一个点所在的坐标系
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            //图像中填入欧几里得深度
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 转换成一维索引，存去畸变之后的激光点
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    //在去畸变的点云fullCloud基础上剔除无效点
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            //提取特征的时候，每一行的前5个和最后5个不考虑
            // 记录每根扫描线起始第5个激光点在一维数组中的索引
            //cloudInfo为lio_sam自定义的msg
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info  激光点距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud     
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
  
        //publishCloud在utility.h头文件中,需要传入发布句柄pubExtractedCloud，提取出的有效点云，该帧时间戳，
        //pubExtractedCloud定义在构造函数中，用来发布去畸变的点云.
        //extractedCloud主要在cloudExtraction中被提取，点云被去除了畸变，
        //另外每行头五个和后五个不要(（仍然被保存，但是之后在提取特征时不要,因为要根据前后五个点算曲率）
        //cloudHeader.stamp 来源于currentCloudMsg,cloudHeader在cachePointCloud中被赋值currentCloudMsg.header
        //而currentCloudMsg是点云队列cloudQueue中提取的
        //lidarFrame:在utility.h中被赋为base_link,
        //在publishCloud函数中，tempCloud.header.frame_id="base_link"(lidarFrame)
        //之后用发布句柄pubExtractedCloud来发布去畸变的点云     
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");                                       //节点初始化

    ImageProjection IP;                                                     //类的实例   因为有构造函数，所以对象一旦被创建就会立马执行
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");           //开始工作

    ros::MultiThreadedSpinner spinner(3);                                   //设置为3线程
    spinner.spin();
    
    return 0;
}
