#ifndef _KIT_HPP_
#define _KIT_HPP_

#include <stdint.h>
#include <vector>

#include <opencv2/opencv.hpp>

typedef class KIT_FOR_BINARY
{
public:
    
    static double distance(const cv::Mat &a,  const cv::Mat &b);

    static void meanValue(const std::vector<cv::Mat> &descriptors, cv::Mat &mean, const std::vector<uint32_t> &mask);

}BIN;

typedef class KIT_FOR_FLOAT
{
public:
    static double distance(const cv::Mat &a,  const cv::Mat &b);

    static void meanValue(const std::vector<cv::Mat> &descriptors, cv::Mat &mean, const std::vector<uint32_t> &mask);

}FLT;

#endif