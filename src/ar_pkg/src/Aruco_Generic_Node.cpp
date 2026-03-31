#include <chrono>
#include <memory>
#include <unordered_map>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

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
#define TOPIC_NAME "ar_marker/webcam"
#define QUEUE_LENGTH 10
#define TIMER_DURATION 50ms

#define MARKER_SIZE_MM 60.0
#define DICTIONARY cv::aruco::DICT_4X4_12
#define DELAY 1
// ---------------------------------------------------------------------

using namespace std::chrono_literals;

// ----------------------------- Helpers --------------------------------

static inline tf2::Quaternion quatFromRvec(const cv::Vec3d & rvec)
{
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

static inline tf2::Quaternion quatSlerp(const tf2::Quaternion & a, const tf2::Quaternion & b, double t)
{
  tf2::Quaternion q1 = a;
  tf2::Quaternion q2 = b;
  q1.normalize();
  q2.normalize();

  double dot = q1.x()*q2.x() + q1.y()*q2.y() + q1.z()*q2.z() + q1.w()*q2.w();
  if (dot < 0.0) {
    q2 = tf2::Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
    dot = -dot;
  }

  const double DOT_THRESHOLD = 0.9995;
  if (dot > DOT_THRESHOLD) {
    tf2::Quaternion out(
      q1.x() + t*(q2.x()-q1.x()),
      q1.y() + t*(q2.y()-q1.y()),
      q1.z() + t*(q2.z()-q1.z()),
      q1.w() + t*(q2.w()-q1.w())
    );
    out.normalize();
    return out;
  }

  const double theta_0 = std::acos(dot);
  const double theta = theta_0 * t;
  const double sin_theta = std::sin(theta);
  const double sin_theta_0 = std::sin(theta_0);

  const double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  const double s1 = sin_theta / sin_theta_0;

  tf2::Quaternion out(
    s0*q1.x() + s1*q2.x(),
    s0*q1.y() + s1*q2.y(),
    s0*q1.z() + s1*q2.z(),
    s0*q1.w() + s1*q2.w()
  );
  out.normalize();
  return out;
}

static inline void quatToEulerRPY(const tf2::Quaternion & q, double & roll, double & pitch, double & yaw)
{
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
}

static inline void drawPoseData(cv::Mat & image, const cv::Point3f & position_mm, const cv::Point3f & eulerAngles)
{
  std::string positionText =
    "Position[mm]: x=" + std::to_string(position_mm.x) +
    " y=" + std::to_string(position_mm.y) +
    " z=" + std::to_string(position_mm.z);

  std::string eulerText =
    "Euler[rpy]: r=" + std::to_string(eulerAngles.x) +
    " p=" + std::to_string(eulerAngles.y) +
    " y=" + std::to_string(eulerAngles.z);

  cv::putText(image, positionText, cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  cv::putText(image, eulerText, cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// ---------------------------------------------------------------------

class PosePublisher : public rclcpp::Node
{
public:
  PosePublisher()
  : Node("pose_publisher_webcam")
  {
    frame_id_ = this->declare_parameter<std::string>("frame_id", "webcam_frame");
    child_prefix_ = this->declare_parameter<std::string>("child_frame_prefix", "aruco_");
    publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
    enable_gui_ = this->declare_parameter<bool>("enable_gui", true);

    filter_enable_ = this->declare_parameter<bool>("filter_enable", true);
    pos_meas_noise_ = this->declare_parameter<double>("pos_meas_noise", 1e-1);
    pos_proc_noise_ = this->declare_parameter<double>("pos_proc_noise", 1e-4);
    quat_alpha_ = this->declare_parameter<double>("quat_smoothing_alpha", 0.25);
    max_jump_mm_ = this->declare_parameter<double>("max_jump_mm", 500.0);

    camera_index_ = this->declare_parameter<int>("camera_index", 0);
    frame_width_ = this->declare_parameter<int>("frame_width", 1280);
    frame_height_ = this->declare_parameter<int>("frame_height", 720);
    marker_size_mm_ = this->declare_parameter<double>("marker_size_mm", MARKER_SIZE_MM);

    // Optional real calibration parameters
    fx_ = this->declare_parameter<double>("fx", 0.0);
    fy_ = this->declare_parameter<double>("fy", 0.0);
    cx_ = this->declare_parameter<double>("cx", 0.0);
    cy_ = this->declare_parameter<double>("cy", 0.0);
    k1_ = this->declare_parameter<double>("k1", 0.0);
    k2_ = this->declare_parameter<double>("k2", 0.0);
    p1_ = this->declare_parameter<double>("p1", 0.0);
    p2_ = this->declare_parameter<double>("p2", 0.0);
    k3_ = this->declare_parameter<double>("k3", 0.0);

    cap_.open(camera_index_);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(this->get_logger(), "Failed to open webcam at index %d", camera_index_);
      throw std::runtime_error("Failed to open webcam");
    }

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);

    // Grab one frame to determine actual size
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Webcam opened but could not read a frame");
      throw std::runtime_error("Failed to read initial frame");
    }

    image_width_ = frame.cols;
    image_height_ = frame.rows;

    // Camera intrinsics:
    // 1) Use supplied calibration if provided
    // 2) Otherwise build a rough approximation from image size
    if (fx_ > 0.0 && fy_ > 0.0) {
      camera_matrix_ = cv::Matx33d::eye();
      camera_matrix_(0, 0) = fx_;
      camera_matrix_(1, 1) = fy_;
      camera_matrix_(0, 2) = cx_;
      camera_matrix_(1, 2) = cy_;
      dist_coeffs_ = cv::Mat(1, 5, CV_64F);
      dist_coeffs_.at<double>(0,0) = k1_;
      dist_coeffs_.at<double>(0,1) = k2_;
      dist_coeffs_.at<double>(0,2) = p1_;
      dist_coeffs_.at<double>(0,3) = p2_;
      dist_coeffs_.at<double>(0,4) = k3_;
      RCLCPP_INFO(this->get_logger(), "Using supplied camera calibration.");
    } else {
      const double f_approx = 0.9 * static_cast<double>(image_width_);
      camera_matrix_ = cv::Matx33d::eye();
      camera_matrix_(0, 0) = f_approx;
      camera_matrix_(1, 1) = f_approx;
      camera_matrix_(0, 2) = static_cast<double>(image_width_) / 2.0;
      camera_matrix_(1, 2) = static_cast<double>(image_height_) / 2.0;
      dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
      RCLCPP_WARN(this->get_logger(),
                  "No calibration provided. Using approximate intrinsics. Pose will be less accurate.");
    }

    dictionary_ = cv::aruco::getPredefinedDictionary(DICTIONARY);
    detector_params_ = cv::aruco::DetectorParameters();

    detector_params_.errorCorrectionRate = 0.0;
    detector_params_.cornerRefinementWinSize = 5;
    detector_params_.cornerRefinementMaxIterations = 30;
    detector_params_.cornerRefinementMinAccuracy = 0.05;

    detector_params_.adaptiveThreshWinSizeMin = 3;
    detector_params_.adaptiveThreshWinSizeMax = 30;
    detector_params_.adaptiveThreshWinSizeStep = 9;

    detector_params_.minMarkerPerimeterRate = 0.01;
    detector_params_.maxMarkerPerimeterRate = 4.0;

    publisher_ = this->create_publisher<ar_pkg::msg::ArucoPoseArray>(TOPIC_NAME, QUEUE_LENGTH);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = this->create_wall_timer(TIMER_DURATION, std::bind(&PosePublisher::publish_pose, this));

    RCLCPP_INFO(this->get_logger(),
                "Webcam ArUco node started. Resolution: %dx%d, marker size: %.2f mm",
                image_width_, image_height_, marker_size_mm_);
  }

private:
  // Webcam
  cv::VideoCapture cap_;
  int camera_index_{0};
  int frame_width_{1280};
  int frame_height_{720};
  int image_width_{0};
  int image_height_{0};

  // Calibration params
  double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};
  double k1_{0.0}, k2_{0.0}, p1_{0.0}, p2_{0.0}, k3_{0.0};

  // OpenCV
  cv::aruco::Dictionary dictionary_;
  cv::aruco::DetectorParameters detector_params_;
  cv::Matx33d camera_matrix_;
  cv::Mat dist_coeffs_;

  std::vector<cv::Vec3d> rvecs_;
  std::vector<cv::Vec3d> tvecs_;
  std::vector<int> ids_;
  std::vector<std::vector<cv::Point2f>> corners_;

  double marker_size_mm_{MARKER_SIZE_MM};

  std::string frame_id_;
  std::string child_prefix_;
  bool publish_tf_{true};
  bool enable_gui_{true};

  bool filter_enable_{true};
  double pos_meas_noise_{1e-1};
  double pos_proc_noise_{1e-4};
  double quat_alpha_{0.25};
  double max_jump_mm_{500.0};

  rclcpp::Publisher<ar_pkg::msg::ArucoPoseArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  struct MarkerFilter {
    cv::KalmanFilter kf;
    bool initialized{false};

    tf2::Quaternion q_filt{0, 0, 0, 1};
    bool q_initialized{false};

    rclcpp::Time last_stamp;
  };

  std::unordered_map<int, MarkerFilter> filters_;

  MarkerFilter & getFilter(int id, double proc_noise, double meas_noise)
  {
    auto & f = filters_[id];
    if (!f.initialized) {
      f.kf = cv::KalmanFilter(6, 3, 0, CV_32F);

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

  static inline void setDt(cv::KalmanFilter & kf, float dt)
  {
    kf.transitionMatrix.at<float>(0,3) = dt;
    kf.transitionMatrix.at<float>(1,4) = dt;
    kf.transitionMatrix.at<float>(2,5) = dt;
  }

  void publish_pose()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Failed to grab frame from webcam");
      return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    corners_.clear();
    ids_.clear();
    rvecs_.clear();
    tvecs_.clear();

    cv::aruco::detectMarkers(gray, dictionary_, corners_, ids_, detector_params_);

    ar_pkg::msg::ArucoPoseArray message;
    const rclcpp::Time stamp = this->get_clock()->now();

    if (!ids_.empty()) {
      cv::aruco::estimatePoseSingleMarkers(
        corners_,
        marker_size_mm_ * 0.001,   // meters
        camera_matrix_,
        dist_coeffs_,
        rvecs_,
        tvecs_);

      if (enable_gui_) {
        cv::aruco::drawDetectedMarkers(frame, corners_, ids_);
      }

      for (size_t i = 0; i < ids_.size(); ++i) {
        const int id = ids_[i];

        // OpenCV pose: tvec is in meters in camera frame
        cv::Point3f pos_mm(
          static_cast<float>(tvecs_[i][0] * 1000.0),
          static_cast<float>(tvecs_[i][1] * 1000.0),
          static_cast<float>(tvecs_[i][2] * 1000.0)
        );

        tf2::Quaternion q_meas = quatFromRvec(rvecs_[i]);

        cv::Point3f pos_out_mm = pos_mm;
        tf2::Quaternion q_out = q_meas;

        if (filter_enable_) {
          auto & f = getFilter(id, pos_proc_noise_, pos_meas_noise_);

          float dt = 0.05f;
          if (f.last_stamp.nanoseconds() > 0) {
            const double dts = (stamp - f.last_stamp).seconds();
            if (dts > 1e-4 && dts < 1.0) {
              dt = static_cast<float>(dts);
            }
          }
          f.last_stamp = stamp;

          setDt(f.kf, dt);

          if (f.kf.statePost.empty() || f.kf.statePost.rows != 6) {
            f.kf.statePost = cv::Mat::zeros(6, 1, CV_32F);
            f.kf.statePost.at<float>(0) = pos_mm.x;
            f.kf.statePost.at<float>(1) = pos_mm.y;
            f.kf.statePost.at<float>(2) = pos_mm.z;
          }

          if (!f.q_initialized) {
            f.q_filt = q_meas;
            f.q_initialized = true;
          }

          cv::Mat pred = f.kf.predict();
          cv::Point3f pred_pos_mm(
            pred.at<float>(0),
            pred.at<float>(1),
            pred.at<float>(2)
          );

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
            pos_out_mm = cv::Point3f(
              est.at<float>(0),
              est.at<float>(1),
              est.at<float>(2)
            );
          } else {
            pos_out_mm = pred_pos_mm;
          }

          const double a = std::clamp(quat_alpha_, 0.0, 1.0);
          f.q_filt = quatSlerp(f.q_filt, q_meas, a);
          q_out = f.q_filt;
        }

        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        quatToEulerRPY(q_out, roll, pitch, yaw);

        ar_pkg::msg::ArucoPose element;
        element.id = id;

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

        if (publish_tf_) {
          geometry_msgs::msg::TransformStamped tfmsg;
          tfmsg.header.stamp = stamp;
          tfmsg.header.frame_id = frame_id_;
          tfmsg.child_frame_id = child_prefix_ + std::to_string(id);

          tfmsg.transform.translation.x = element.position.x * 0.001;
          tfmsg.transform.translation.y = element.position.y * 0.001;
          tfmsg.transform.translation.z = element.position.z * 0.001;

          tfmsg.transform.rotation.x = q_out.x();
          tfmsg.transform.rotation.y = q_out.y();
          tfmsg.transform.rotation.z = q_out.z();
          tfmsg.transform.rotation.w = q_out.w();

          tf_broadcaster_->sendTransform(tfmsg);
        }

        if (enable_gui_) {
          cv::aruco::drawAxis(
            frame,
            camera_matrix_,
            dist_coeffs_,
            rvecs_[i],
            tvecs_[i],
            static_cast<float>(marker_size_mm_ * 0.5 * 0.001)
          );

          cv::Point3f eul(
            static_cast<float>(roll),
            static_cast<float>(pitch),
            static_cast<float>(yaw)
          );
          drawPoseData(frame, pos_out_mm, eul);
        }
      }

      if (enable_gui_) {
        cv::imshow("Detected Markers and Pose", frame);
        cv::waitKey(DELAY);
      }
    } else if (enable_gui_) {
      cv::imshow("Detected Markers and Pose", frame);
      cv::waitKey(DELAY);
    }

    publisher_->publish(message);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PosePublisher>());
  rclcpp::shutdown();
  return 0;
}
