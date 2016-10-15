#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>

class FeatureProcessor
{
public:

    //@brief enum for choosing keypoint extractor
  enum class KE_TYPE
  {
    SIFT,
    FAST,
    SURF
  };
    //@brief enum for choosing descriptor computer
  enum class DC_TYPE
  {
    SIFT,
    FREAK,
    SURF
  };

  FeatureProcessor(int kps_per_basket = 5,
                   int dist_threshold = 30,
                   int baskets_per_dimension = 6,
                   FeatureProcessor::KE_TYPE ke = KE_TYPE::FAST,
                   FeatureProcessor::DC_TYPE dc = DC_TYPE::FREAK);

    //@brief main worker - compares photos and makes matches
  void next_frame(const cv::Mat& img);

    //@brief remove last frame
  void remove_frame();

    //@brief get kps from penultimate photo
  const std::vector<cv::KeyPoint>& get_old_kps()
  {
    return *old_kps_;
  }
    //@brief get kps from tha last photo
  const std::vector<cv::KeyPoint>& get_new_kps()
  {
    return *new_kps_;
  }
    //@brief get matches
    //  match.queryIdx - new kps index
    //  match.trainIdx - old kps index
  const std::vector<cv::DMatch>& get_matches()
  {
    return matches_;
  }

    //==  getters for parameters
  int get_kps_per_basket()
  {
    return kps_per_basket_;
  }
  int get_baskets_per_dimension()
  {
    return baskets_per_dimension_;
  }
  int get_dist_threshold()
  {
    return dist_threshold_;
  }
  KE_TYPE get_ke_type()
  {
    return ke_type_;
  }
  DC_TYPE get_dc_type()
  {
    return dc_type_;
  }
    //==

    //== setters for parameters
  void set_kps_per_basket(int kps_per_basket)
  {
    if (kps_per_basket <= 0)
      throw std::runtime_error("bad args");
    kps_per_basket_ = kps_per_basket;
  }
  void set_baskets_per_dimension(int baskets_per_dimension)
  {
    if (baskets_per_dimension <= 0)
      throw std::runtime_error("bad args");
    baskets_per_dimension_ = baskets_per_dimension;
  }
  void set_dist_threshold(int dist_threshold)
  {
    if (dist_threshold <= 0)
      throw std::runtime_error("bad args");
    dist_threshold_ = dist_threshold;
  }
  void set_ke_type(KE_TYPE ke_type);
  void set_dc_type(DC_TYPE dc_type);
    //==

private:
  void detect_kps(const cv::Mat& i_img,
                  std::vector<cv::KeyPoint>& i_kps);

  void compute_dc(const cv::Mat& img,
                  std::vector<cv::KeyPoint>& kps,
                  cv::Mat& des);

    //== workers pointers
  cv::Ptr<cv::Feature2D> ke_cv;
  cv::Ptr<cv::DescriptorExtractor> dc_cv;

  cv::Ptr<cv::DescriptorMatcher> matcher_;
    //==

    //== parameters
  KE_TYPE ke_type_;
  DC_TYPE dc_type_;

  int kps_per_basket_;
  int baskets_per_dimension_;
  int dist_threshold_;
    //==

    //== buffers
  std::shared_ptr<std::vector<cv::KeyPoint> > old_kps_;
  std::shared_ptr<cv::Mat> old_des_;

  std::shared_ptr<std::vector<cv::KeyPoint> > new_kps_;
  std::shared_ptr<cv::Mat> new_des_;

  std::vector<cv::DMatch> matches_;
    //==
};