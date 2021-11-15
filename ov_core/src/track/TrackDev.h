#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"

namespace ov_core {

/**
 * @brief Use tracked of features from device
 *
 */
class TrackDev : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param binocular if we should do binocular feature tracking or stereo if there are multiple cameras
   */
  explicit TrackDev(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, bool binocular)
	: TrackBase(cameras, 10, 10, binocular, HistogramMethod::NONE) {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message);

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
