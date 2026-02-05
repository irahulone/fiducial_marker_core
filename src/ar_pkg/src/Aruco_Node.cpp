// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <memory>
#include <unordered_map>
#include <string>
#include <numeric>
#include <array>
#include <cmath>
#include <algorithm>

#include "utils.hpp"
#include "aruco.hpp"

#include "rclcpp/rclcpp.hpp"
#include "ar_pkg/msg/aruco_pose_array.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_ros/transform_broadcaster.h"

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

// ----------------------------- CONFIG --------------------------------
#define TOPIC_NAME "ar_marker/zed"
#define QUEUE_LENGTH 10
#define TIMER_DURATION 50ms

// Default path: point-cloud based pose + orientation vectors (your custom)
#define ZED_POSE true

#define CONFIDENCE_THRESHOLD          95
#define TEXTURE_CONFIDENCE_THRESHOLD  95
#define MINIMUM_DEPTH_DISTANCE        50    // In camera units (mm)
#define ERROR_CORRECTION_RATE         0
#define MARKER_SIZE                   60    // mm
#define DELAY                         1

#define DICTIONARY                    cv::aruco::DICT_4X4_12

#define CAMERA_RESOLUTION             sl::RESOLUTION::HD1080
#define CAMERA_UNITS                  sl::UNIT::MILLIMETER
#define DEPTH_MODE                    sl::DEPTH_MODE::NEURAL
#define ENABLE_SENSORS                true
#define ENABLE_DEPTH_STABILIZATION    true
#define ENABLE_FILL_MODE              true
#define ENABLE_IMU_FUSION             true
#define SET_GRAVITY_AS_ORIGIN         true
#define ENABLE_AREA_MEMORY            true
#define ENABLE_POSE_SMOOTHING         true
#define ENABLE_SET_AS_STATIC          true
#define PLANE_ORIENTATION             false
// ---------------------------------------------------------------------

using namespace std::chrono_literals;
using std::placeholders::_1;

// ----------------------------- Helpers --------------------------------

static inline tf2::Quaternion quatFromRvec(const cv::Vec3d& rvec) {
  // Rodrigues -> rotation matrix
  cv::Mat Rcv;
  cv::Rodrigues(rvec, Rcv);

  tf2::Matrix3x3 R(
    Rcv.at<double>(0,0), Rcv.at<double>(0,1), Rcv.at<double>(0,2),
    Rcv.at<double>(1,0), Rcv.at<double>(1,1), Rcv.at<double>(1,2),
    Rcv.at<double>(2,0), Rcv.at<double>(2,1), Rcv.at<double>(2,2)
  );

  tf2::Quaternion q;
  R.getRotation(q);
  q.normalize();
  return q;
}

static inline tf2::Quaternion quatFromYourMarkerVectors(const std::array<cv::Point3f, 3>& markerVectors) {
  // Your getMarkerVector() returns 3 basis vectors (assumed orthonormal-ish).
  // We'll build a 3x3 rotation matrix (columns = basis) in your convention.
  // NOTE: If your markerVectors encode rows instead of columns, swap accordingly.
  const cv::Point3f& ex = markerVectors[0];
  const cv::Point3f& ey = markerVectors[1];
  const cv::Point3f& ez = markerVectors[2];

  tf2::Matrix3x3 R(
    ex.x, ey.x, ez.x,
    ex.y, ey.y, ez.y,
    ex.z, ey.z, ez.z
  );

  tf2::Quaternion q;
  R.getRotation(q);
  q.normalize();
  return q;
}

static inline tf2::Quaternion quatSlerp(const tf2::Quaternion& a, const tf2::Quaternion& b, double t) {
  // tf2 doesn't ship a direct slerp in all builds; implement minimal slerp.
  tf2::Quaternion q1 = a;
  tf2::Quaternion q2 = b;
  q1.normalize();
  q2.normalize();

  double dot = q1.x()*q2.x() + q1.y()*q2.y() + q1.z()*q2.z() + q1.w()*q2.w();
  if (dot < 0.0) { // shortest path
    q2 = tf2::Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
    dot = -dot;
  }

  const double DOT_THRESHOLD = 0.9995;
  if (dot > DOT_THRESHOLD) {
    // almost linear
    tf2::Quaternion out(
      q1.x() + t*(q2.x()-q1.x()),
      q1.y() + t*(q2.y()-q1.y()),
      q1.z() + t*(q2.z()-q1.z()),
      q1.w() + t*(q2.w()-q1.w())
    );
    out.normalize();
    return out;
  }

  double theta_0 = std::acos(dot);
  double theta = theta_0 * t;
  double sin_theta = std::sin(theta);
  double sin_theta_0 = std::sin(theta_0);

  double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  double s1 = sin_theta / sin_theta_0;

  tf2::Quaternion out(
    (s0*q1.x()) + (s1*q2.x()),
    (s0*q1.y()) + (s1*q2.y()),
    (s0*q1.z()) + (s1*q2.z()),
    (s0*q1.w()) + (s1*q2.w())
  );
  out.normalize();
  return out;
}

static inline void quatToEulerRPY(const tf2::Quaternion& q, double& roll, double& pitch, double& yaw) {
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
}

// Minimal pose overlay (optional)
static inline void drawPoseData(cv::Mat& image, const cv::Point3f& position, const cv::Point3f& eulerAngles) {
  std::string positionText = "Position: x=" + std::to_string(position.x) +
                             " y=" + std::to_string(position.y) +
                             " z=" + std::to_string(position.z);
  std::string eulerText = "Euler(rpy): r=" + std::to_string(eulerAngles.x) +
                          " p=" + std::to_string(eulerAngles.y) +
                          " y=" + std::to_string(eulerAngles.z);

  cv::putText(image, positionText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  cv::putText(image, eulerText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// ---------------------------------------------------------------------

class PosePublisher : public rclcpp::Node
{
public:
  PosePublisher()
  : Node("pose_publisher")
  {
    // ---------------- ROS Params (so you can fix frames without recompiling) -------------
    frame_id_     = this->declare_parameter<std::string>("frame_id", "zed_left_camera_frame");
    child_prefix_ = this->declare_parameter<std::string>("child_frame_prefix", "aruco_");
    publish_tf_   = this->declare_parameter<bool>("publish_tf", true);
    enable_gui_   = this->declare_parameter<bool>("enable_gui", false);

    filter_enable_ = this->declare_parameter<bool>("filter_enable", true);
    pos_meas_noise_ = this->declare_parameter<double>("pos_meas_noise", 1e-1);
    pos_proc_noise_ = this->declare_parameter<double>("pos_proc_noise", 1e-4);
    quat_alpha_ = this->declare_parameter<double>("quat_smoothing_alpha", 0.25);
    max_jump_mm_ = this->declare_parameter<double>("max_jump_mm", 500.0);
    use_plane_orientation_ = this->declare_parameter<bool>("use_plane_orientation", false);

    // ------------------------------------------------------------------------------------

    init_params.camera_resolution = CAMERA_RESOLUTION;
    init_params.coordinate_units = CAMERA_UNITS;
    init_params.sensors_required = ENABLE_SENSORS;

    // IMPORTANT:
    // Your original code used LEFT_HANDED_Y_UP.
    // We'll keep it, and publish TF in a frame that explicitly matches it
    // unless you change it here.
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;

    init_params.depth_minimum_distance = MINIMUM_DEPTH_DISTANCE;
    init_params.depth_stabilization = ENABLE_DEPTH_STABILIZATION;
    init_params.depth_mode = DEPTH_MODE;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "Error, unable to open ZED camera: " << sl::toString(err).c_str() << "\n";
      zed.close();
      std::exit(EXIT_FAILURE);
    }

    runtime_params.confidence_threshold = CONFIDENCE_THRESHOLD;
    runtime_params.texture_confidence_threshold = TEXTURE_CONFIDENCE_THRESHOLD;
    runtime_params.enable_fill_mode = ENABLE_FILL_MODE;

    auto cameraInfo = zed.getCameraInformation();
    image_size = cameraInfo.camera_configuration.resolution;

    image_zed.alloc(image_size, sl::MAT_TYPE::U8_C4);
    point_cloud.alloc(image_size, sl::MAT_TYPE::F32_C4);

    // Wrap ZED image in cv::Mat (CPU pointer)
    image_ocv = cv::Mat(image_zed.getHeight(), image_zed.getWidth(), CV_8UC4,
                        image_zed.getPtr<sl::uchar1>(sl::MEM::CPU));

    // Intrinsics
    auto calibInfo = cameraInfo.camera_configuration.calibration_parameters.left_cam;
    camera_matrix = cv::Matx33d::eye();
    camera_matrix(0, 0) = calibInfo.fx;
    camera_matrix(1, 1) = calibInfo.fy;
    camera_matrix(0, 2) = calibInfo.cx;
    camera_matrix(1, 2) = calibInfo.cy;

    dist_coeffs = cv::Vec4f::zeros();

    dictionary = cv::aruco::getPredefinedDictionary(DICTIONARY);

    // OpenCV aruco older API: plain struct, no create(), no cornerRefinementMethod
    parameters.errorCorrectionRate = ERROR_CORRECTION_RATE;

    parameters.cornerRefinementWinSize = 5;
    parameters.cornerRefinementMaxIterations = 30;
    parameters.cornerRefinementMinAccuracy = 0.05;

    parameters.adaptiveThreshWinSizeMin = 3;
    parameters.adaptiveThreshWinSizeMax = 30;
    parameters.adaptiveThreshWinSizeStep = 9;

    parameters.minMarkerPerimeterRate = 0.01;
    parameters.maxMarkerPerimeterRate = 4.0;

    actual_marker_size_meters = (MARKER_SIZE * 0.001); // meters
    std::cout << "ArUco marker size = " << actual_marker_size_meters << " m (" << MARKER_SIZE << " mm)\n";

    // ZED tracking
    tracking_params.enable_imu_fusion = ENABLE_IMU_FUSION;
    tracking_params.set_gravity_as_origin = SET_GRAVITY_AS_ORIGIN;
    tracking_params.enable_area_memory = ENABLE_AREA_MEMORY;
    tracking_params.enable_pose_smoothing = ENABLE_POSE_SMOOTHING;
    tracking_params.set_as_static = ENABLE_SET_AS_STATIC;

    err = zed.enablePositionalTracking(tracking_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "Error enabling positional tracking: " << sl::toString(err) << std::endl;
      zed.close();
      std::exit(EXIT_FAILURE);
    }

    publisher_ = this->create_publisher<ar_pkg::msg::ArucoPoseArray>(TOPIC_NAME, QUEUE_LENGTH);

    // TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = this->create_wall_timer(TIMER_DURATION, std::bind(&PosePublisher::publish_pose, this));
  }

private:
  // ZED
  sl::Camera zed;
  sl::Rotation rotation_matrix;
  sl::Mat image_zed;
  sl::Mat point_cloud;
  sl::InitParameters init_params;
  sl::RuntimeParameters runtime_params;
  sl::Resolution image_size;
  sl::PositionalTrackingParameters tracking_params;

  // OpenCV
  cv::Mat image_ocv;
  cv::Mat image_ocv_rgb;
  cv::aruco::Dictionary dictionary;
  cv::Matx<float, 4, 1> dist_coeffs;
  cv::Matx33d camera_matrix;

  std::vector<cv::Vec3d> rvecs, tvecs;
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;

  double actual_marker_size_meters;

  std::string frame_id_;
  std::string child_prefix_;
  bool publish_tf_{true};
  bool enable_gui_{false};

  bool filter_enable_{true};
  double pos_meas_noise_{1e-1};
  double pos_proc_noise_{1e-4};
  double quat_alpha_{0.25};
  double max_jump_mm_{500.0};
  bool use_plane_orientation_{false};

  rclcpp::Publisher<ar_pkg::msg::ArucoPoseArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // ---------------------- Filtering State per marker -------------------
  struct MarkerFilter {
    // 6-state: x y z vx vy vz, 3 measurement: x y z
    cv::KalmanFilter kf;
    bool initialized{false};

    tf2::Quaternion q_filt{0,0,0,1};
    bool q_initialized{false};

    rclcpp::Time last_stamp;
  };

  std::unordered_map<int, MarkerFilter> filters_;

  MarkerFilter& getFilter(int id, double proc_noise, double meas_noise) {
    auto& f = filters_[id];
    if (!f.initialized) {
      f.kf = cv::KalmanFilter(6, 3, 0, CV_32F);

      // State: [x y z vx vy vz]
      // Transition: constant velocity with dt filled at runtime
      f.kf.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);

      f.kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
      f.kf.measurementMatrix.at<float>(0,0) = 1.f;
      f.kf.measurementMatrix.at<float>(1,1) = 1.f;
      f.kf.measurementMatrix.at<float>(2,2) = 1.f;

      cv::setIdentity(f.kf.processNoiseCov, cv::Scalar::all(proc_noise));
      cv::setIdentity(f.kf.measurementNoiseCov, cv::Scalar::all(meas_noise));
      cv::setIdentity(f.kf.errorCovPost, cv::Scalar::all(1e-1));

      f.initialized = true;
    }
    return f;
  }

  static inline void setDt(cv::KalmanFilter& kf, float dt) {
    // constant velocity model
    kf.transitionMatrix.at<float>(0,3) = dt;
    kf.transitionMatrix.at<float>(1,4) = dt;
    kf.transitionMatrix.at<float>(2,5) = dt;
  }

  // ---------------------- Main loop -----------------------------------
  void publish_pose()
  {

    sl::Pose camera_pose;

    if (zed.grab(runtime_params) != sl::ERROR_CODE::SUCCESS) {
      return;
    }

    // Grab pose (world) for your getCorrectedPosition call
    zed.getPosition(camera_pose, sl::REFERENCE_FRAME::WORLD);
    rotation_matrix = sl::Rotation(camera_pose.getRotationMatrix());
    
    // Retrieve image + point cloud
    zed.retrieveImage(image_zed, sl::VIEW::LEFT, sl::MEM::CPU, image_size);
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU, image_size);

    // Convert BGRA -> BGR -> Gray
    cv::cvtColor(image_ocv, image_ocv_rgb, cv::COLOR_BGRA2BGR);
    cv::Mat grayImage;
    cv::cvtColor(image_ocv_rgb, grayImage, cv::COLOR_BGR2GRAY);

    // IMPORTANT: clear vectors every frame to avoid any carry-over surprises
    corners.clear();
    ids.clear();
    rvecs.clear();
    tvecs.clear();

    // Detect
    cv::aruco::detectMarkers(grayImage, dictionary, corners, ids, parameters);

    // Prepare outgoing message
    ar_pkg::msg::ArucoPoseArray message;

    if (!ids.empty()) {
      // OpenCV pose estimation (used for orientation (and position if ZED_POSE=false))
      cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_size_meters, camera_matrix, dist_coeffs, rvecs, tvecs);

      // Optional display
      if (enable_gui_) {
        cv::aruco::drawDetectedMarkers(image_ocv_rgb, corners, ids);
      }

      const rclcpp::Time stamp = this->get_clock()->now();

      for (size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];

        // ------------------- Compute position -------------------
        // Center pixel
        cv::Point2f center(0.f, 0.f);
        for (const auto& p : corners[i]) center += p;
        center *= (1.0f / 4.0f);

        cv::Point3f pos_mm;

        if (ZED_POSE) {
          // Your pipeline: point cloud -> corrected position
          cv::Point3f point3D = retrieveValid3DPoint(
            static_cast<int>(std::round(center.x)),
            static_cast<int>(std::round(center.y)),
            point_cloud
          );

          // "Corrected" using camera/world rotation (your helper)
          point3D = getCorrectedPosition(point3D, rotation_matrix);

          // point3D is in ZED units (mm) and ZED coordinate system (left-handed Y up per init_params)
          // Keep it in that frame; do NOT axis-swap hacks here.
          pos_mm = point3D;
        } else {
          // OpenCV tvec is in meters in camera frame
          pos_mm = cv::Point3f(
            static_cast<float>(tvecs[i][0] * 1000.0),
            static_cast<float>(tvecs[i][1] * 1000.0),
            static_cast<float>(tvecs[i][2] * 1000.0)
          );
        }

        // ------------------- Compute orientation -------------------
        tf2::Quaternion q_meas(0,0,0,1);

        // Option A: plane orientation (NOT marker orientation) if you really want it
        if (use_plane_orientation_) {
          sl::Plane plane;
          const auto status = zed.findPlaneAtHit(
            sl::uint2(static_cast<int>(std::round(center.x)), static_cast<int>(std::round(center.y))),
            plane
          );
          if (status == sl::ERROR_CODE::SUCCESS) {
            sl::Orientation o = plane.getPose().getOrientation();
            // sl::Orientation indexing: (x,y,z,w) accessible via operator()
            q_meas = tf2::Quaternion(o(0), o(1), o(2), o(3));
            q_meas.normalize();
          } else {
            // fallback to OpenCV aruco pose
            q_meas = quatFromRvec(rvecs[i]);
          }
        } else {
          if (ZED_POSE) {
            // Your custom orientation from point cloud vectors
            std::array<cv::Point3f, 3> markerVectors = getMarkerVector(corners[i], zed, point_cloud, rotation_matrix);
            q_meas = quatFromYourMarkerVectors(markerVectors);
          } else {
            // Correct conversion (rvec is NOT Euler)
            q_meas = quatFromRvec(rvecs[i]);
          }
        }

        // ------------------- Filter / Sanity checks -------------------
        cv::Point3f pos_out_mm = pos_mm;
        tf2::Quaternion q_out = q_meas;

        if (filter_enable_) {
          auto& f = getFilter(id, pos_proc_noise_, pos_meas_noise_);

          float dt = 0.05f;
          if (f.initialized && f.last_stamp.nanoseconds() > 0) {
            const double dts = (stamp - f.last_stamp).seconds();
            if (dts > 1e-4 && dts < 1.0) dt = static_cast<float>(dts);
          }
          f.last_stamp = stamp;

          setDt(f.kf, dt);

          // If first time, initialize state to measurement
          if (f.kf.statePost.empty() || f.kf.statePost.rows != 6) {
            f.kf.statePost = cv::Mat::zeros(6, 1, CV_32F);
          }

          if (!f.q_initialized) {
            f.q_filt = q_meas;
            f.q_initialized = true;
          }

          // Predict
          cv::Mat pred = f.kf.predict();
          cv::Point3f pred_pos_mm(
            pred.at<float>(0), pred.at<float>(1), pred.at<float>(2)
          );

          // Reject insane jumps (often depth failures)
          const double dx = pos_mm.x - pred_pos_mm.x;
          const double dy = pos_mm.y - pred_pos_mm.y;
          const double dz = pos_mm.z - pred_pos_mm.z;
          const double jump = std::sqrt(dx*dx + dy*dy + dz*dz);

          if (jump < max_jump_mm_) {
            cv::Mat meas(3, 1, CV_32F);
            meas.at<float>(0) = pos_mm.x;
            meas.at<float>(1) = pos_mm.y;
            meas.at<float>(2) = pos_mm.z;

            cv::Mat est = f.kf.correct(meas);
            pos_out_mm = cv::Point3f(est.at<float>(0), est.at<float>(1), est.at<float>(2));
          } else {
            // keep predicted position
            pos_out_mm = pred_pos_mm;
          }

          // Quaternion smoothing (slerp)
          const double a = std::clamp(quat_alpha_, 0.0, 1.0);
          f.q_filt = quatSlerp(f.q_filt, q_meas, a);
          q_out = f.q_filt;
        }

        // Euler for message (roll/pitch/yaw), if your msg expects it
        // NOTE: this is "RPY" of the quaternion, not Rodrigues rvec.
        double roll=0, pitch=0, yaw=0;
        quatToEulerRPY(q_out, roll, pitch, yaw);

        // ------------------- Fill ROS message -------------------
        ar_pkg::msg::ArucoPose element;
        element.id = id;

        // Keep in the ZED camera frame coordinates as-is (mm)
        // If you want REP103/optical frame conversions, do it via TF/static transforms.
        element.position.x = pos_out_mm.x;
        element.position.y = pos_out_mm.y;
        element.position.z = pos_out_mm.z;

        element.orientation.euler.x = roll;
        element.orientation.euler.y = pitch;
        element.orientation.euler.z = yaw;

        element.orientation.quaternion.i = q_out.x();
        element.orientation.quaternion.j = q_out.y();
        element.orientation.quaternion.k = q_out.z();
        element.orientation.quaternion.w = q_out.w();

        message.data.push_back(element);

        // ------------------- TF publish -------------------
        if (publish_tf_) {
          geometry_msgs::msg::TransformStamped tfmsg;
          tfmsg.header.stamp = stamp;
          tfmsg.header.frame_id = frame_id_;

          tfmsg.child_frame_id = child_prefix_ + std::to_string(id);

          // meters in TF
          tfmsg.transform.translation.x = element.position.x * 0.001;
          tfmsg.transform.translation.y = element.position.y * 0.001;
          tfmsg.transform.translation.z = element.position.z * 0.001;

          tfmsg.transform.rotation.x = q_out.x();
          tfmsg.transform.rotation.y = q_out.y();
          tfmsg.transform.rotation.z = q_out.z();
          tfmsg.transform.rotation.w = q_out.w();

          tf_broadcaster_->sendTransform(tfmsg);
        }

        // Optional overlay of one marker (example)
        if (enable_gui_) {
          cv::Point3f eul(static_cast<float>(roll), static_cast<float>(pitch), static_cast<float>(yaw));
          drawPoseData(image_ocv_rgb, pos_out_mm, eul);
        }
      } // for each id

      if (enable_gui_) {
        cv::imshow("Detected Markers and Pose", image_ocv_rgb);
        cv::waitKey(DELAY);
      }
    }

    publisher_->publish(message);
  }

  // Aruco params must be a Ptr in newer OpenCV
  cv::aruco::DetectorParameters parameters;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PosePublisher>());
  rclcpp::shutdown();
  return 0;
}
