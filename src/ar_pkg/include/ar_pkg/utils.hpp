#pragma once

#include <cmath>
#include <array>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#define MAX_POINT_RADIUS 							10

cv::Point3f getCorrectedPosition(const cv::Point3f& position, const sl::Rotation& rotation_matrix){
    cv::Point3f corrected_position;
    corrected_position.x = rotation_matrix.r00 * position.x + rotation_matrix.r01 * position.y + rotation_matrix.r02 * position.z;
    corrected_position.y = rotation_matrix.r10 * position.x + rotation_matrix.r11 * position.y + rotation_matrix.r12 * position.z;
    corrected_position.z = rotation_matrix.r20 * position.x + rotation_matrix.r21 * position.y + rotation_matrix.r22 * position.z;
    
    return corrected_position;
}

// Function to print sl::Rotation
void printRotation(const sl::Rotation& rotation) {
    std::cout << "Rotation Matrix:" << std::endl;
    std::cout << rotation.r00 << " " << rotation.r01 << " " << rotation.r02 << std::endl;
    std::cout << rotation.r10 << " " << rotation.r11 << " " << rotation.r12 << std::endl;
    std::cout << rotation.r20 << " " << rotation.r21 << " " << rotation.r22 << std::endl;
}

// Function to convert sl::Rotation to Euler angles (roll, pitch, yaw) for left-handed Y-up coordinate system
void printRotationAsEulerAngles(const sl::Rotation& rotation) {
    // Extract elements from the rotation matrix
    float r00 = rotation.r00;  // X-axis to the right
    float r01 = rotation.r01;  // Y-axis up
    float r02 = rotation.r02;  // Z-axis forward
    float r10 = rotation.r10;
    float r11 = rotation.r11;
    float r12 = rotation.r12;
    float r20 = rotation.r20;
    float r21 = rotation.r21;
    float r22 = rotation.r22;

    // Calculate sy to check for gimbal lock and other calculations
    float sy = std::sqrt(r00 * r00 + r20 * r20); // Combination of X and Z components

    bool is_singular = sy < 1e-6; // If sy is close to zero, we are in a gimbal lock scenario

    float roll, pitch, yaw;
    if (!is_singular) {
        roll = std::atan2(r12, r11);   // Roll around the Z-axis
        pitch = std::atan2(-r20, sy);  // Pitch around the X-axis
        yaw = std::atan2(r10, r00);    // Yaw around the Y-axis
    } else {
        roll = std::atan2(-r21, r22);  // Alternative roll in case of gimbal lock
        pitch = std::atan2(-r20, sy);  // Pitch remains the same
        yaw = 0;                       // Yaw is undefined in gimbal lock scenario
    }

    // Convert from radians to degrees for easier interpretation
    roll = roll * 180.0 / M_PI;
    pitch = pitch * 180.0 / M_PI;
    yaw = yaw * 180.0 / M_PI;

    // Print Euler angles
    std::cout << "Euler Angles (degrees):" << std::endl;
    std::cout << "Roll: " << roll << std::endl;
    std::cout << "Pitch: " << pitch << std::endl;
    std::cout << "Yaw: " << yaw << std::endl;
}


// Function to convert quaternion to Euler angles using xyzw format
cv::Point3f quaternionToEuler(sl::Orientation& q) {
  cv::Point3f euler;

  // Extract the components of the quaternion in xyzw format
  double x = q.x;  // The x component
  double y = q.y;  // The y component
  double z = q.z;  // The z component
  double w = q.w;  // The w (scalar) component

  // Roll (x-axis rotation)
  double sinr_cosp = 2 * (w * x + y * z);
  double cosr_cosp = 1 - 2 * (x * x + y * y);
  euler.x = std::atan2(sinr_cosp, cosr_cosp);

  // Pitch (y-axis rotation)
  double sinp = 2 * (w * y - z * x);
  if (std::abs(sinp) >= 1)
      euler.y = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
      euler.y = std::asin(sinp);

  // Yaw (z-axis rotation)
  double siny_cosp = 2 * (w * z + x * y);
  double cosy_cosp = 1 - 2 * (y * y + z * z);
  euler.z = std::atan2(siny_cosp, cosy_cosp);

  return euler;
}

// Function to convert radians to degrees
inline double radToDeg(const double radians) {
  return radians * (180.0 / M_PI);
}

void printArrayAsVectors(const std::array<cv::Point3f, 4>& points) {
  std::cout << "Vector form: [";
  for (size_t i = 0; i < points.size(); ++i) {
    std::cout << "[" << points[i].x << ", " << points[i].y << ", " << points[i].z << "]";
    if (i < points.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

inline cv::Point2f addPoints(const cv::Point2f& a, const cv::Point2f& b) {
  return cv::Point2f(a.x + b.x, a.y + b.y);
}

// Function to calculate the vector from P1 to P2
inline cv::Point3f calculateVector(const cv::Point3f& P1, const cv::Point3f& P2) {return cv::Point3f(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z);}

// Function to normalize a vector
cv::Point3f normalizeVector(const cv::Point3f& vector) {
	double magnitude = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);

	if (magnitude == 0) {
		std::cerr << "Cannot normalize a zero-length vector." << std::endl;
		return vector; // Return the original vector unchanged
	}
	return cv::Point3f(vector.x / magnitude, vector.y / magnitude, vector.z / magnitude);
}

// Function to retrieve a valid 3D point from the point cloud
cv::Point3f retrieveValid3DPoint(int x, int y, sl::Mat& pointCloud) {
	sl::float4 pointCloudValue;

	// Check the original point
	pointCloud.getValue(x, y, &pointCloudValue);
	if (!std::isnan(pointCloudValue.x) && !std::isnan(pointCloudValue.y) && !std::isnan(pointCloudValue.z)) {
		return cv::Point3f(pointCloudValue.x, pointCloudValue.y, pointCloudValue.z);
	}

	// Circular search for a valid neighboring point
	for (int radius = 1; radius <= MAX_POINT_RADIUS; ++radius) {
		for (int dx = -radius; dx <= radius; ++dx) {
			for (int dy = -radius; dy <= radius; ++dy) {
				if (std::sqrt(dx * dx + dy * dy) <= radius) { // Ensure the search is within the current radius
					int newX = x + dx;
					int newY = y + dy;

					// Ensure the new coordinates are within the image bounds
					if (newX >= 0 && newX < pointCloud.getWidth() && newY >= 0 && newY < pointCloud.getHeight()) {
						pointCloud.getValue(newX, newY, &pointCloudValue);
						if (!std::isnan(pointCloudValue.x) && !std::isnan(pointCloudValue.y) && !std::isnan(pointCloudValue.z)) {
							return cv::Point3f(pointCloudValue.x, pointCloudValue.y, pointCloudValue.z);
						}
					}
				}
			}
		}
	}

	// If no valid point found, return a point with NaN values to indicate failure
	return cv::Point3f(NAN, NAN, NAN);
}


bool isArrayNan(const std::array<cv::Point3f, 3>& arr) {
  for (const auto& point : arr) {
    if (!std::isnan(point.x) || !std::isnan(point.y) || !std::isnan(point.z)) {
      return false;
    }
  }
  return true;
}

bool isArrayEmpty(const std::array<cv::Point3f, 3>& arr) {
  for (const auto& point : arr) {
    if (point != cv::Point3f(0, 0, 0)) {
      return false;
    }
  }
  return true;
}


std::array<cv::Point3f, 3> calculateRotationalVector(std::array<cv::Point3f, 4>& points3D){

  std::array<cv::Point3f, 3> markerVectors;
  // Function to check if a point is valid (not NaN)
  auto isValid = [](const cv::Point3f& point) {
    return !std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z);
  };

  // Variables to store the valid vectors
  cv::Point3f vectorX1, vectorY1, vectorX2, vectorY2, vectorZ;

  // Check if all corners are valid
  if (isValid(points3D[0]) && isValid(points3D[1]) && isValid(points3D[2]) && isValid(points3D[3])) {
    // Calculate vectors using first pair of opposite corners
    vectorX1 = calculateVector(points3D[0], points3D[1]); // X: Forward direction
    vectorY1 = -1 * calculateVector(points3D[0], points3D[3]); // Y: Left direction

    // Calculate vectors using second pair of opposite corners
    vectorX2 = -1 * calculateVector(points3D[2], points3D[3]); // X: Forward direction
    vectorY2 = calculateVector(points3D[2], points3D[1]); // Y: Left direction

    // Average the vectors
    vectorX1 = (vectorX1 + vectorX2) * 0.5;
    vectorY1 = (vectorY1 + vectorY2) * 0.5;
  } else {
    // Handle cases where not all corners are valid, similar to the previous approach
    if (isValid(points3D[0]) && isValid(points3D[1]) && isValid(points3D[3])) {
      vectorX1 = calculateVector(points3D[0], points3D[1]); // X: Forward direction
      vectorY1 = -1 * calculateVector(points3D[0], points3D[3]); // Y: Left direction
    } else if (isValid(points3D[1]) && isValid(points3D[2]) && isValid(points3D[0])) {
      vectorX1 = -1 * calculateVector(points3D[1], points3D[2]); // X: Forward direction
      vectorY1 = -1 * calculateVector(points3D[1], points3D[0]); // Y: Left direction
    } else if (isValid(points3D[2]) && isValid(points3D[3]) && isValid(points3D[1])) {
      vectorX1 = -1 * calculateVector(points3D[2], points3D[3]); // X: Forward direction
      vectorY1 = calculateVector(points3D[2], points3D[1]); // Y: Left direction
    } else if (isValid(points3D[3]) && isValid(points3D[0]) && isValid(points3D[2])) {
      vectorX1 = calculateVector(points3D[3], points3D[0]); // X: Forward direction
      vectorY1 = calculateVector(points3D[3], points3D[2]); // Y: Left direction
    } else {
      return std::array<cv::Point3f, 3>();
    }
  }

  // Normalize vectorX
  vectorX1 = normalizeVector(vectorX1);

  // Orthogonalize vectorY to vectorX
  vectorY1 = vectorY1 - vectorX1 * (vectorY1.dot(vectorX1) / vectorX1.dot(vectorX1));

  // Normalize vectorY
  vectorY1 = normalizeVector(vectorY1);

  // Calculate the Z vector (up direction) using the cross product of X and Y
  vectorZ = vectorX1.cross(vectorY1);
  vectorZ = normalizeVector(vectorZ);

  // Store the orthogonalized vectors
  markerVectors[0] = vectorX1; // X-axis: Forward
  markerVectors[1] = vectorY1; // Y-axis: Left
  markerVectors[2] = vectorZ; // Z-axis: Up

  return markerVectors;
}

// Function to retrieve 3D points from 2D corners and calculate normalized vectors
std::array<cv::Point3f, 3> getMarkerVector(const std::vector<cv::Point2f>& corners, sl::Camera& zed, sl::Mat& pointCloud, const sl::Rotation& rotation_matrix) {
  std::array<cv::Point3f, 4> points3D;

  // Ensure that the point cloud has data
  if (!pointCloud.isInit()) {
    throw std::runtime_error("Point cloud not initialized.");
  }

  // Retrieve the 3D points for all four corners
  for (size_t i = 0; i < 4; ++i) {
    points3D[i] = retrieveValid3DPoint(std::round(corners[i].x), std::round(corners[i].y), pointCloud);
    points3D[i] = getCorrectedPosition(points3D[i], rotation_matrix);
  }

  return calculateRotationalVector(points3D);
}


// Function to retrieve 3D points from 2D corners and calculate normalized vectors
std::array<cv::Point3f, 4> getCornerPosition(const std::vector<cv::Point2f>& corners, sl::Mat& pointCloud) {
  std::array<cv::Point3f, 4> points3D;

  // Ensure that the point cloud has data
  if (!pointCloud.isInit()) {
    throw std::runtime_error("Point cloud not initialized.");
  }

  // Retrieve the 3D points for all four corners
  for (size_t i = 0; i < 4; ++i) {
    points3D[i] = retrieveValid3DPoint(std::round(corners[i].x), std::round(corners[i].y), pointCloud);
  }

  return points3D;
}


// Function to extract Euler angles from a rotation matrix (XYZ convention)
cv::Point3f rotationMatrixToEulerAngles(const std::array<cv::Point3f, 3>& R) {
  cv::Point3f eulerAngles;

  // Extract the Pitch (Y-axis rotation)
  eulerAngles.y = std::asin(-R[0].z); // pitch (θ)
  
  if (std::cos(eulerAngles.y) != 0) {
    // Extract the Roll (X-axis rotation)
    eulerAngles.x = std::atan2(R[1].z, R[2].z); // roll (φ)
    // Extract the Yaw (Z-axis rotation)
    eulerAngles.z = std::atan2(R[0].y, R[0].x); // yaw (ψ)
  } else {
    // Gimbal lock case
    eulerAngles.x = std::atan2(-R[2].y, R[1].y); // roll (φ)
    eulerAngles.z = 0; // yaw (ψ) is not uniquely defined
  }

  return eulerAngles;
}

// Function to convert rotation matrix to quaternion (XYZ convention)
std::array<double, 4> rotationMatrixToQuaternion(const std::array<cv::Point3f, 3>& R) {
	std::array<double, 4> q; // Quaternion: [w, x, y, z]

	double trace = R[0].x + R[1].y + R[2].z;

	if (trace > 0) {
		double S = std::sqrt(trace + 1.0f) * 2; // S = 4 * q.w
		q[0] = 0.25f * S; // w
		q[1] = (R[2].y - R[1].z) / S; // x
		q[2] = (R[0].z - R[2].x) / S; // y
		q[3] = (R[1].x - R[0].y) / S; // z
	} else if ((R[0].x > R[1].y) && (R[0].x > R[2].z)) {
		double S = std::sqrt(1.0f + R[0].x - R[1].y - R[2].z) * 2; // S = 4 * q.x
		q[0] = (R[2].y - R[1].z) / S; // w
		q[1] = 0.25f * S; // x
		q[2] = (R[0].y + R[1].x) / S; // y
		q[3] = (R[0].z + R[2].x) / S; // z
	} else if (R[1].y > R[2].z) {
		double S = std::sqrt(1.0f + R[1].y - R[0].x - R[2].z) * 2; // S = 4 * q.y
		q[0] = (R[0].z - R[2].x) / S; // w
		q[1] = (R[0].y + R[1].x) / S; // x
		q[2] = 0.25f * S; // y
		q[3] = (R[1].z + R[2].y) / S; // z
	} else {
		double S = std::sqrt(1.0f + R[2].z - R[0].x - R[1].y) * 2; // S = 4 * q.z
		q[0] = (R[1].x - R[0].y) / S; // w
		q[1] = (R[0].z + R[2].x) / S; // x
		q[2] = (R[1].z + R[2].y) / S; // y
		q[3] = 0.25f * S; // z
	}

	return q;
}


cv::Point3f calculateCentroid(const std::array<cv::Point3f, 4>& points) {
  cv::Point3f centroid(0, 0, 0);
  for (const auto& point : points) {
    centroid += point;
  }
  centroid /= 4.0f;
  return centroid;
}

cv::Mat estimatePlaneNormal(const std::array<cv::Point3f, 4>& points, cv::Point3f& centroid) {

    // Step 1: Construct matrix A for SVD using all four points
    cv::Mat A(4, 3, CV_32F);  // 4 rows (one for each point), 3 columns (x, y, z differences)
    for (int i = 0; i < 4; ++i) {
        A.at<float>(i, 0) = points[i].x - centroid.x;
        A.at<float>(i, 1) = points[i].y - centroid.y;
        A.at<float>(i, 2) = points[i].z - centroid.z;
    }

    // Step 2: Perform SVD
    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt);

    // The normal of the plane is the third row of vt
    cv::Mat normal = vt.row(2).t();
    return normal;  // Return the normal vector
}

std::array<cv::Point3f, 4> fitSquareToPoints(const std::array<cv::Point3f, 4>& points, float side_length, cv::Point3f& centroid) {
  // Step 1: Calculate the centroid of the points
  //cv::Point3f centroid = calculateCentroid(points);

  // Step 2: Estimate the plane normal (or average normal)
  cv::Mat normal = estimatePlaneNormal(points, centroid);

  // Step 3: Choose two perpendicular vectors on the plane
  cv::Mat u = normal.cross(cv::Mat(cv::Point3f(1, 0, 0)));
  if (cv::norm(u) == 0) u = normal.cross(cv::Mat(cv::Point3f(0, 1, 0)));
  u = u / cv::norm(u);

  cv::Mat v = normal.cross(u);
  u = u * (side_length / 2.0f);
  v = v * (side_length / 2.0f);

  // Step 4: Generate the square centered at the centroid in a clockwise order
  std::array<cv::Point3f, 4> square;
  square[0] = centroid + cv::Point3f(u) + cv::Point3f(v);  // Top-left
  square[1] = centroid + cv::Point3f(u) - cv::Point3f(v);  // Top-right
  square[2] = centroid - cv::Point3f(u) - cv::Point3f(v);  // Bottom-right
  square[3] = centroid - cv::Point3f(u) + cv::Point3f(v);  // Bottom-left

  // Step 4: Find the closest square corner to each point in the input and reorder
  std::array<cv::Point3f, 4> orderedSquare;
  std::vector<bool> used(4, false);

  for (int i = 0; i < 4; ++i) {
    float minDistance = std::numeric_limits<float>::max();
    int closestIndex = -1;

    for (int j = 0; j < 4; ++j) {
      if (!used[j]) {
        float distance = cv::norm(points[i] - square[j]);
        if (distance < minDistance) {
          minDistance = distance;
          closestIndex = j;
        }
      }
    }

    orderedSquare[i] = square[closestIndex];
    used[closestIndex] = true;  // Mark this point as used
  }

  return orderedSquare;
}
