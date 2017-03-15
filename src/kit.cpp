#include "kit.hpp"


double BIN::distance(const cv::Mat &a,  const cv::Mat &b)
{
    assert(a.type()==CV_8U);
    assert(a.rows==1);

    // Bit count function got from:
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

    const uint64_t *pa, *pb;
    pa = a.ptr<uint64_t>(0); // a & b are actually CV_8U
    pb = b.ptr<uint64_t>(0);

    uint64_t v, ret = 0;
    for(size_t i = 0; i < a.cols / sizeof(uint64_t); ++i, ++pa, ++pb)
    {
        v = *pa ^ *pb;
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &
         (uint64_t)~(uint64_t)0/15*3);
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
        ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>
         (sizeof(uint64_t) - 1) * CHAR_BIT;
    }

    return ret;
}

void BIN::meanValue(const std::vector<cv::Mat> &descriptors, cv::Mat &mean, const std::vector<uint32_t> &mask)
{

    if(descriptors.empty())
    {
        mean.release();
        return;
    }
    else if(descriptors.size() == 1)
    {
        mean = descriptors[0].clone();
    }
    else 
    {
        //! for ORB
        uint32_t cols = descriptors[0].cols;
        uint32_t ndim = cols * CHAR_BIT;
        int* sum = new int[ndim];
        std::fill(sum, sum+ndim, 0);

        uint32_t count_num = 0;
        if(mask.empty())
        {   
            count_num = descriptors.size();

            for(size_t i = 0; i < count_num; ++i)
            {
                const cv::Mat &d = descriptors[i];
                const unsigned char *p = d.ptr<unsigned char>(0);

                for(int j = 0; j < d.cols; ++j, ++p, sum+=8)
                {
                    if(*p & (1 << 7)) ++sum[0];
                    if(*p & (1 << 6)) ++sum[1];
                    if(*p & (1 << 5)) ++sum[2];
                    if(*p & (1 << 4)) ++sum[3];
                    if(*p & (1 << 3)) ++sum[4];
                    if(*p & (1 << 2)) ++sum[5];
                    if(*p & (1 << 1)) ++sum[6];
                    if(*p & (1))      ++sum[7];
                }
                sum-=ndim;
            }

        }
        else
        {
            count_num = mask.size();

            for(size_t i = 0; i < count_num; ++i)
            {
                const cv::Mat &d = descriptors[mask[i]];
                const unsigned char *p = d.ptr<unsigned char>(0);

                for(int j = 0; j < d.cols; ++j, ++p, sum+=8)
                {
                    if(*p & (1 << 7)) ++sum[0];
                    if(*p & (1 << 6)) ++sum[1];
                    if(*p & (1 << 5)) ++sum[2];
                    if(*p & (1 << 4)) ++sum[3];
                    if(*p & (1 << 3)) ++sum[4];
                    if(*p & (1 << 2)) ++sum[5];
                    if(*p & (1 << 1)) ++sum[6];
                    if(*p & (1))      ++sum[7];
                }
                sum-=ndim;
            }

        }


        mean = cv::Mat::zeros(1, cols, CV_8U);
        unsigned char *p = mean.ptr<unsigned char>(0);

        const int N2 = (int)count_num / 2 + count_num % 2;
        for(uint32_t i = 0; i < ndim; ++i)
        {
            if(sum[i] >= N2)
            {
                // set bit
                *p |= 1 << (7 - (i % 8));
            }

            if(i % 8 == 7) ++p;
        }
    }
}

double FLT::distance(const cv::Mat &a,  const cv::Mat &b)
{
    assert(a.type()==CV_32F);
    assert(a.rows==1);

    double sqd = 0.;
    const float *a_ptr=a.ptr<float>(0);
    const float *b_ptr=b.ptr<float>(0);
    for(int i = 0; i < a.cols; i ++)
        sqd += (a_ptr[i] - b_ptr[i])*(a_ptr[i] - b_ptr[i]);
    return sqd;
}

void FLT::meanValue(const std::vector<cv::Mat> &descriptors, cv::Mat &mean, const std::vector<uint32_t> &mask)
{
    if(descriptors.empty())
    {
        mean.release();
        return;
    }
    else if(descriptors.size() == 1)
    {
        mean = descriptors[0].clone();
    }
    else 
    {
        uint32_t ndim = descriptors[0].cols;
        double* sum = new double[ndim];
        std::fill(sum, sum+ndim, 0.0);

        uint32_t count_num = 0;
        if(mask.empty())
        {   
            count_num = descriptors.size();

            for(size_t i = 0; i < count_num; ++i)
            {
                const cv::Mat &d = descriptors[i];
                const float *p = d.ptr<float>(0);

                for(int j = 0; j < ndim; ++j, ++p)
                {
                    sum[j] += *p;
                }
            }

        }
        else
        {
            count_num = mask.size();

            for(size_t i = 0; i < count_num; ++i)
            {
                const cv::Mat &d = descriptors[mask[i]];
                const float *p = d.ptr<float>();

                for(int j = 0; j < ndim; ++j, ++p)
                {
                    sum[j] += *p;
                }
            }

        }

        mean = cv::Mat::zeros(1, ndim, CV_32F);
        float *p = mean.ptr<float>();

        for(uint32_t i = 0; i < ndim; ++i)
        {
            *(p+i) = sum[i]/count_num;
        }
    }
}
