#include <string>
#include <vector>
#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/types.hpp>

#include <feature_processor/feature_processor.h>

  //#define DEBUG_MODE

FeatureProcessor::FeatureProcessor(int kps_per_basket,
                                   int dist_threshold,
                                   int baskets_per_dimension,
                                   FeatureProcessor::KE_TYPE ke_type,
                                   FeatureProcessor::DC_TYPE dc_type):
matcher_(new cv::BFMatcher())
{
  set_kps_per_basket(kps_per_basket);
  set_baskets_per_dimension(baskets_per_dimension);
  set_dist_threshold(dist_threshold);
  set_dc_type(dc_type);
  set_ke_type(ke_type);
}

void FeatureProcessor::set_ke_type(FeatureProcessor::KE_TYPE ke_type)
{
  ke_type_ = ke_type;

  ke_cv.release();

  switch (ke_type_)
  {
    case KE_TYPE::SIFT:
      ke_cv = cv::xfeatures2d::SiftFeatureDetector::create();
      break;
    case KE_TYPE::FAST:
      ke_cv = cv::FastFeatureDetector::create();
      break;
    case KE_TYPE::SURF:
      ke_cv = cv::xfeatures2d::SurfFeatureDetector::create();
      break;
    default:
      throw std::runtime_error("unknown ke");
  }
}

void FeatureProcessor::set_dc_type(FeatureProcessor::DC_TYPE dc_type)
{
  dc_type_ = dc_type;

  dc_cv.release();

  switch (dc_type_)
  {
    case DC_TYPE::SIFT:
      dc_cv = cv::xfeatures2d::SiftDescriptorExtractor::create();
      break;
    case DC_TYPE::FREAK:
      dc_cv = cv::xfeatures2d::FREAK::create();
      break;
    case DC_TYPE::SURF:
      dc_cv = cv::xfeatures2d::SurfDescriptorExtractor::create();
      break;
    default:
      throw std::runtime_error("unknown dc");
  }
}

void FeatureProcessor::next_frame(const cv::Mat &img)
{
  matches_.clear();
#ifdef DEBUG_MODE
  static int counter(0);
  ++counter;
#endif

    //== if zero iteration just init old data
  if (!old_kps_)
  {
    old_kps_ =  std::make_shared<std::vector<cv::KeyPoint> >();
    old_des_ = std::make_shared<cv::Mat>();
    detect_kps(img, *old_kps_);

#ifdef DEBUG_MODE
    cv::Mat debug_img;
    cv::drawKeypoints(img, *old_kps_, debug_img, cv::Scalar(0, 0, 255));
    cv::imwrite("detection" + std::to_string(counter) + ".png", debug_img);
#endif

    compute_dc(img, *old_kps_, *old_des_);
    return;
  }
    //==

    //== if NOT first iteration - need to shift
  if (new_kps_) {
    old_kps_ = new_kps_;
    old_des_ = new_des_;
  }
    //==

    //== detect and compute for new photo
  new_kps_ = std::make_shared<std::vector<cv::KeyPoint> >();
  new_des_ = std::make_shared<cv::Mat>();
  detect_kps(img, *new_kps_);

#ifdef DEBUG_MODE
  cv::Mat debug_img;
  cv::drawKeypoints(img, *new_kps_, debug_img, cv::Scalar(0, 0, 255));
  cv::imwrite("detection" + std::to_string(counter) + ".png", debug_img);
#endif

  compute_dc(img, *new_kps_, *new_des_);
    //==

    //== match
  if ((new_kps_->size()) && (old_kps_->size()))
  {
    matcher_->match(*new_des_, *old_des_, matches_);
  }
    //==

    //== dist filter
  for (int i = matches_.size()-1; i >= 0; --i)
  {
    double dist = cv::norm((*new_kps_)[matches_[i].queryIdx].pt -
                           (*old_kps_)[matches_[i].trainIdx].pt);
    if (dist > dist_threshold_)
    {
      matches_.erase(matches_.begin() + i);
    }
  }
    //==
}

void FeatureProcessor::remove_frame()
{
  new_kps_.reset();
  new_des_.reset();
}

void FeatureProcessor::compute_dc(const cv::Mat &img,
                                  std::vector<cv::KeyPoint> &kps,
                                  cv::Mat &des)
{
  if (dc_cv)
  {
    dc_cv->compute(img, kps, des);
    return;
  }
}
void FeatureProcessor::detect_kps(const cv::Mat &i_img,
                                  std::vector<cv::KeyPoint> &i_kps)
{
  int height_step = i_img.size().height / baskets_per_dimension_;
  int width_step = i_img.size().width / baskets_per_dimension_;
  std::vector<cv::KeyPoint> buff;

  for (int i = 0; i < baskets_per_dimension_; ++i) {
    for (int j = 0; j < baskets_per_dimension_; ++j) {
      cv::Range height_range(i * height_step, (i+1) * height_step);
      cv::Range width_range(j * width_step, (j+1) * width_step);
      cv::Mat basket_img = i_img(height_range, width_range);

      if (ke_cv) {
        ke_cv->detect(basket_img, buff);
      }

      cv::KeyPointsFilter::retainBest(buff, kps_per_basket_);

      for(auto& kp: buff){
        kp.pt.x += j * width_step;
        kp.pt.y += i * height_step;
      }
      i_kps.insert(i_kps.end(), buff.begin(), buff.end());
    }
  }
}