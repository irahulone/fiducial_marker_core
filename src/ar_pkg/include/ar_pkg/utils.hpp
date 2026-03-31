#pragma once

#include <cmath>
#include <array>
#include <limits>
#include <opencv2/opencv.hpp>

inline cv::Point2f addPoints(const cv::Point2f& a, const cv::Point2f& b) {
  return cv::Point2f(a.x + b.x, a.y + b.y);
}

inline cv::Point3f calculateVector(const cv::Point3f& P1, const cv::Point3f& P2) {
  return cv::Point3f(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z);
}

inline cv::Point3f normalizeVector(const cv::Point3f& vector) {
  double magnitude = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
  if (magnitude == 0.0) {
    return vector;
  }
  return cv::Point3f(vector.x / magnitude, vector.y / magnitude, vector.z / magnitude);
}

inline double radToDeg(const double radians) {
  return radians * (180.0 / M_PI);
}
