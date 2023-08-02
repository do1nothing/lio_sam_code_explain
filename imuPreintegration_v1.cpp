#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
//语法上：使用了GTSAM的符号简写功能
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
/*
功能：
    订阅激光里程计(与IMUPreintegration订阅的有所不同)和imu里程计(来自IMUPreintegration)
    根据前一时刻激光里程计，以及这一时刻到当前时刻imu里程计变换增量，计算当前时刻imu里程计
    发布lio_sam/imu/path用于局部地图展示
*/
class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;                                     //互斥锁

    ros::Subscriber subImuOdometry;                     //来自IMUPreintegration
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;                          //根据节点关系图可以得出这是用于rviz展示
    //仿射变换一般表示4*4矩阵
    Eigen::Affine3f lidarOdomAffine;                    //保存雷达里程计数据    位姿变换矩阵
    Eigen::Affine3f imuOdomAffineFront;                 //距离雷达里程计最近imu里程计位姿变换矩阵   （这里有个小瑕疵：这里仅定义了，第二处用的时候又定义加初始化一次）
    Eigen::Affine3f imuOdomAffineBack;                  //表示当前时刻imu里程计数据吗？？？
    //坐标系转换时候使用
    tf::TransformListener tfListener;                   //用于订阅和查询不同坐标系之间的变换关系
    tf::StampedTransform lidar2Baselink;                //激光雷达坐标系到机器人底盘坐标系之间的变换关系，并记录了变换的时间戳

    double lidarOdomTime = -1;                          //雷达里程计时间
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        // 如果lidar系与baselink系不同（激光系和载体系），需要外部提供二者之间的变换关系
        // 目的就是得到lidar2Baselink，后边要用
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                //最新时刻， 等待3s               
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                // lidar系到baselink系的变换    查询两个坐标系之间的变换关系，并将结果存储在 lidar2Baselink 变量中
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            //如果查询失败，则会抛出 tf::TransformException 异常，并将错误信息记录在 ex 变量中，程序会将错误信息通过 ROS_ERROR 打印出来。
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅imu里程计topic name: odometry/imu_incremental
        //odometry/imu_incremental是增量内容，即两帧激光里程计之间的预积分内容,加上开始的激光里程计本身有的位姿
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }
    /*
        odom变换矩阵
        功能：ROS->PCL变换矩阵
    */
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        //从odom提取位置和姿态
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);                //将姿态信息转换为旋转矩阵
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);           //将旋转和平移信息转换成变换矩阵
    }

    /*
        Lidar里程计回调函数
        功能：锁数据，记录Lidar里程计变换矩阵及时间戳
    */
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        //激光里程计对应变换矩阵
        lidarOdomAffine = odom2affine(*odomMsg);// 将雷达里程计信息（坐标、欧拉角）从ROS->TF
        // 激光里程计时间戳
        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    /*
        imu里程计回调函数
        功能：计算当前时刻imu里程计位姿；发布里程计位姿与里程计路径（用于rviz展示）
    */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // map与odom设为同一个系        这东西在哪里会用？建图用？   通过广播，该变换矩阵的关系将被其他节点接收和使用。
        static tf::TransformBroadcaster tfMap2Odom;                     //广播变换矩阵
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));//表示map与odom坐标系重合
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));//发布

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);// 记录通过imu估计的雷达里程计信息(后面简称imu里程计信息)

        // get latest odometry (at current IMU stamp)
        // 当没有订阅到最终优化后的里程计信息时，直接返回
        if (lidarOdomTime == -1)
            return;
        // 当订阅到最终优化后的lidar里程计信息时，剔除掉比该帧还老的imu里程计信息帧
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front()); // 最近的一帧激光里程计时刻对应imu里程计位姿
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());// 当前时刻imu里程计位姿
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;// front时刻与back时刻之间imu里程计增量位姿变换
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;//当前时刻imu里程计位姿=最近的一帧激光里程计位姿 * imu里程计增量位姿变换
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);// 通过仿射变换提取出位置信息和用欧拉角表示的姿态信息
        
        // publish latest odometry
        // 话题名"odometry/imu"
        // 在激光的基础上加上增量
        //根据当前时刻imu里程计位姿得到当前时刻雷达里程计位姿并发布（明明发布的imu，起个变量名非得叫laserOdometry）
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 发布最新的odom与base_link之间的转换关系，为了rviz显示imu里程计路径用
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发布imu对应的路径信息            只是最近一帧激光里程计时刻与当前时刻之间的一段
        //滑动窗口
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;          //再次执行时，last_path_time不再为1
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 每隔0.1s添加一个
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            // 删除最近一帧激光里程计时刻之前的imu里程计
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};
/*
功能：
    用两帧lidar里程计之间的imu预积分量构建因子图，优化当前帧(关键帧)的状态（位姿、速度、偏执）
    以优化后的状态为基础，添加imu预积分量，得到每一时刻imu里程计
*/
class IMUPreintegration : public ParamServer
{
public:
    //交叉锁
    std::mutex mtx;

    ros::Subscriber subImu;                         //原始imu数据
    ros::Subscriber subOdometry;                    //雷达里程计
    ros::Publisher pubImuOdometry;                  //imu增量，发布给TransformFusion类

    bool systemInitialized = false;                                     //重置参数时会改为false
    //噪声协方差        用于因子图优化使用
    /*
    作用

    */
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;             //先验位姿噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;              //先验旋转速度噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;             //先验偏差噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;            //噪声修正  下同
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    //预积分器
    //负责预积分两个激光里程计之间的imu数据，作为约束加入因子图，并且优化出bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    //用来根据新的激光里程计到达后已经优化好的bias，预测从当前帧开始，下一帧激光里程计到达之前的imu里程计增量
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
    //给imuIntegratorOpt_提供数据来源，不要的就弹出(从队头开始出发，比当前激光里程计数据早的imu通通积分，用一个扔一个)
    std::deque<sensor_msgs::Imu> imuQueOpt;//原始imu数据
    std::deque<sensor_msgs::Imu> imuQueImu;
    //imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;
    //imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;
    // ISAM2优化器              Incremental smoothing and mapping
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;   //总的因子图模型
    gtsam::Values graphValues;                  //因子图的值

    const double delta_t = 0;

    int key = 1;
    

    // imu-lidar位姿变换
    //这点要注意，这只是一个平移变换，因为旋转部分是单位矩阵
    //同样头文件的imuConverter中，也只有一个旋转变换。这里绝对不可以理解为把imu数据转到lidar下的变换矩阵。
    //事实上，作者后续是把imu数据先用imuConverter旋转到雷达系下（但其实还差了个平移）。
    //作者真正是把雷达数据又根据lidar2Imu反向平移了一下，和转换以后差了个平移的imu数据在“中间系”对齐
    //之后算完又从中间系通过imu2Lidar挪回了雷达系进行publish
    //这里注意一点就是，雷达系始终没有进行旋转，仅仅只是平移。extTrans来自imu和雷达的外参标定
    //这里是定义了两个变量，后续会用到
    // T_bl: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));


    IMUPreintegration()
    {
        //订阅imu原始数据
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        //订阅lidar里程计
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        //发布imu里程计，提供给TransformFusion类
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);
        //imu预积分的噪声协方差     看来和重力有关
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        //imuAccNoise和imuGyrNoise来自params.yaml，即imu内参标定结果
        //计算结果依旧是3*3矩阵             三维单位矩阵*噪声的平方
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias 6维向量
        //先验噪声
        //note:Diagnal表示对角线矩阵，Isotropic表示向量
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s           方差大表明置信度低
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        //偏差来自于imu，bias
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        //用于预测每一时刻imu里程计     包含标定的noise，初值为0的bias
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        //用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /*
    功能：
        1.每隔100帧激光里程计，重置ISAM2优化器，添加里程计、速度、偏置先验因子，执行优化
        2.计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计
        添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前状态
        3.优化之后，执行重传播；优化更新了imu的偏置，用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);//锁线程

        double currentCorrectionTime = ROS_TIME(odomMsg);//获取到雷达odom的时间戳

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;
        // 当前帧激光位姿，来自scan-to-map匹配、因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        //位姿是否发生退化
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;//这个协方差是什么含义？
                                                                                //主对角线值越小表明测量误差越小
        //格式转化为gtsam的位姿
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));



        //主要的工作是更新imu的时间戳，然后将lidar的先验位姿等添加到因子图中，初始化一些参数等
        // 0. initialize system
        // 矫正过程的初始化     第一帧
        if (systemInitialized == false)
        {
            resetOptimization();// 重置isam2优化器及非线性因子图

            // pop old IMU message
            // 丢弃老的imu信息  从imu优化队列中删除当前帧激光里程计时刻之前的imu数据
            //不停的把队列头部的时间戳进行更新，直到大于矫正时间
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            // 通过最终优化过的雷达位姿初始化先验的位姿信息并添加到因子图中
            // lidar->imu
            prevPose_ = lidarPose.compose(lidar2Imu);//6维
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);//X（0）为索引；prevPose_为先验位姿；priorPoseNoise是协方差矩阵
            graphFactors.add(priorPose);//将先验信息添加到因子图中
            // initial velocity
            // 初始化先验速度信息为0并添加到因子图中
            prevVel_ = gtsam::Vector3(0, 0, 0);                     //3维
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);//构建一个三维速度向量的先验约束
            graphFactors.add(priorVel);
            // initial bias
            // 初始化先验偏置信息为0并添加到因子图中
            prevBias_ = gtsam::imuBias::ConstantBias();//6维
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);//构建先验偏置约束
            graphFactors.add(priorBias);
            // add values
            // 设置变量的初始估计值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 将因子图更新到isam2优化器中
            optimizer.update(graphFactors, graphValues);
            //将参数清空
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            //resetIntegrationAndSetBias()是GTSAM库中IMU预积分器（IMU integrator）类ImuIntegrator的一个方法。
            //resetIntegrationAndSetBias()方法的作用是重置预积分器的状态并设置新的偏置值。在实际应用中，由于IMU漂移会随时间不断增加，因此需要定期重置预积分器的状态以避免积分误差的累积。
            //resetIntegrationAndSetBias()方法可以在IMU偏置发生变化时调用，用于更新预积分器的状态。
            //具体来说，该方法会重新设置预积分器的姿态和速度，并将新的偏置值设置为预积分器的当前偏置值，从而保证预积分器的状态与IMU当前状态一致。
            
            key = 1;
            systemInitialized = true;       //第一帧结束
            return;
        }

        //获取最新的协方差，重置因子图，由当前协方差构建新的先验约束，类似与上面的初始化先验信息
        // reset graph for speed
        // 当isam2规模太大时，进行边缘化，重置优化器和因子图
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            resetOptimization();
            // 按最新关键帧的协方差将位姿、速度、偏置因子添加到因子图中
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            // 并用最新关键帧的位姿、速度、偏置初始化对应的因子
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 并将最新初始化的因子图更新到重置的isam2优化器中
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1; // 重置关键帧数量
        }


        // 1. integrate imu data and optimize

        //主要是获取两帧IMU数据之间的时间差dt，然后存储，并更新上一帧的时间戳，然后在队列中移除
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            // 对相邻两次优化之间的imu帧进行积分，并移除
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                //计算预积分量和雅可比矩阵
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        //构建约束
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);//由degenerate选择合适的噪声偏置
        graphFactors.add(pose_factor);
        // insert predicted values  根据上一时刻姿态和bias预测当前时刻imu积分值
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        // 将最新关键帧相关的因子图更新到isam2优化器中，并进行优化
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 获取当前关键帧的优化结果，并将结果置为先前值
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        // 当前约束任务完成，复位预积分约束，同时设置下一次预积分bias
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // 对优化结果进行失败检测: 当速度和偏置太大时，则认为优化失败
        // 评判的标准是：(1)prevVel_的范数大于30，或prevBias_的(2)acc或(3)gyr范数大于1.0，则认为优化失败，重置参数重新开始优化。
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 2. 优化后，重新对imu里程计进行预积分
        // 利用优化结果更新prev状态      
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        // 注意以下操作针对的是另一个imu队列
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
    // 按理说上边的while都把imuQueImu时间小于currentCorrectionTime的pop完了，下边进行的是当前激光帧之后imu数据，这一步与imuHander的工作不久重复了吗？
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 将优化后的imu偏置信息更新到预积分器内
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 从矫正时间(lidarodom)开始，对imu数据重新进行预积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(
                                                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }
    /*
    订阅imu原始数据
    1、用上一帧激光里程计时刻对应的状态、偏置,施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态，也就是imu里程计
    2、imu里程计位姿转到lidar系，发布坐标、四元数；imu坐标系下线速度、角速度（已修正）
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)//指向imu类型的常量指针
    {
        //防止imu队列和偏差队列被雷达里程计线程修改
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);//将imu信息转换到雷达坐标系下表达,其实也就是获得雷达运动的加速度、角速度和姿态信息
        //将IMU数据压入队列中
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);
        //判断是否进行预积分                odometryHandler()最后会doneFirstOpt=true
        if (doneFirstOpt == false)
            return;
        
        double imuTime = ROS_TIME(&thisImu);
        // 获取相邻两帧imu数据时间差，
        //没有获取到上一帧时间戳dt=1/500
        //否则取两帧的时间差
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu); 
        lastImuT_imu = imuTime;//将当前帧时间变为上一帧

        // integrate this single imu message
        // imu预积分器添加一帧imu数据，注：这个预积分器的起始时刻是上一帧激光里程计时刻
        imuIntegratorImu_->integrateMeasurement(
                                                gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态
        //在GTSAM中，predict()函数通常用于生成预测因子，然后与实际测量因子进行比较和优化，以计算最终的状态估计。
        //prevStateOdom和prevBiasOdom都来自于odometryHandler()
        //prevStateOdom包含位姿和速度
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish imu_odometry
        
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;       //来自另一个类
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);// 平移
        
        // 发布通过估计的雷达里程计信息(后面都称为imu里程计信息)
        //雷达坐标系(重合)
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        //雷达坐标系(差一个平移)
        odometry.twist.twist.linear.x = currentState.velocity().x();        //直接优化好的？
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x(); //加了偏执才算优化好的？
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);//包含了位置、旋转四元数、线速度、角速度
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
