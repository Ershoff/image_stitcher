#include <string>
#include <vector>
#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/types.hpp>

#include <feature_processor/feature_processor.h>

static std::string filename("/Users/lev/iitp/prj_vrhels/record_2.mp4");
static std::string out_filename("out_record.avi");

static int scale = 2;
  //#define DEBUG_MODE

int video_with_kps()
{
    //== prepair
  cv::VideoCapture raw_video(filename);
  if(!raw_video.isOpened())
    throw std::runtime_error("cant open video");
  int frame_count = raw_video.get(cv::CAP_PROP_FRAME_COUNT);

  cv::Size scaled_frame_size((int)raw_video.get(CV_CAP_PROP_FRAME_WIDTH),
                             (int)raw_video.get(CV_CAP_PROP_FRAME_HEIGHT));
  scaled_frame_size /= scale;

  cv::VideoWriter video_output;
  video_output.open(out_filename,
                    CV_FOURCC('M','J','P','G'),
                    raw_video.get(cv::CAP_PROP_FPS),
                    scaled_frame_size * scale);
  if(!video_output.isOpened())
    throw std::runtime_error("cant open video out");

  FeatureProcessor processor;
  processor.set_ke_type(FeatureProcessor::KE_TYPE::FAST);
  processor.set_dc_type(FeatureProcessor::DC_TYPE::FREAK);

#ifdef DEBUG_MODE
  cv::Mat old_photo;
#endif
    //==

    //== loop over all frames
  for (int i = 0; i < 20; ++i)
  {
#ifdef DEBUG_MODE
    if(!(i % 10)) std::cout << i << std::endl;
#endif
    cv::Mat new_photo;
    raw_video >> new_photo;
    cv::cvtColor(new_photo, new_photo, cv::COLOR_RGB2GRAY);
    cv::resize(new_photo, new_photo, scaled_frame_size);

    processor.next_frame(new_photo);

    std::vector<cv::KeyPoint> good_kps;

    for(auto& match: processor.get_matches()) {
        //i_kps[match.queryIdx].size = 15; <- for better visualisation
      good_kps.push_back(processor.get_new_kps()[match.queryIdx]);
    }
#ifdef DEBUG_MODE
    if (i) {
      cv::Mat out;
      cv::drawMatches(new_photo, processor.get_new_kps(), old_photo, processor.get_old_kps(), processor.get_matches(), out);
      cv::imwrite(std::to_string(i) + ".png", out);
    }
    old_photo = new_photo;
#endif
    cv::drawKeypoints(new_photo, good_kps, new_photo, cv::Scalar(0, 0, 255));
    cv::resize(new_photo, new_photo, scaled_frame_size * scale);
    video_output << new_photo;
  }
  video_output.release();
  return 0;
}

int main()
{
  video_with_kps();
}