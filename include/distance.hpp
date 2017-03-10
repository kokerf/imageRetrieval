#ifndef _DISTANCE_HPP_
#define _DISTANCE_HPP_

class Distance
{
public:
    //Distance();
    //~Distance();
    
    double distance(const cv::Mat &a,  const cv::Mat &b)
    {
        if (a.type()==CV_8U){

            // Bit count function got from:
            // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
            // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

            const uint64_t *pa, *pb;
            pa = a.ptr<uint64_t>(); // a & b are actually CV_8U
            pb = b.ptr<uint64_t>();

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
        else{
            double sqd = 0.;
            assert(a.type()==CV_32F);
            assert(a.rows==1);
            const float *a_ptr=a.ptr<float>(0);
            const float *b_ptr=b.ptr<float>(0);
            for(int i = 0; i < a.cols; i ++)
                sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
            return sqd;
        }
    }

};

#endif