#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

#include "akmeans.hpp"

using std::string;


#ifdef WIN32
void loadImages(const string &file_directory, std::vector<string> &image_filenames)
{
    _finddata_t file;
    intptr_t fileHandle;
    std::string filename = file_directory + "\\*.jpg";
    fileHandle = _findfirst(filename.c_str(), &file);
    if (fileHandle  != -1L) {
        do {
            std::string image_name = file_directory + "\\" + file.name;
            image_filenames.push_back(image_name);
            std::cout << image_name << std::endl;
        } while (_findnext(fileHandle, &file) == 0);
    }

    _findclose(fileHandle);
    std::sort(image_filenames.begin(), image_filenames.end());
}
#else
void loadImages(const std::string &strFileDirectory, std::vector<string> &vstrImageFilenames)
{
    DIR* dir = opendir(strFileDirectory.c_str());
    dirent* p = NULL;
    while((p = readdir(dir)) != NULL)
    {  
        if(p->d_name[0] != '.')
        {  
            std::string imageFilename = strFileDirectory + string(p->d_name);
            vstrImageFilenames.push_back(imageFilename);
            //cout << imageFilename << endl;
        } 
    } 

    closedir(dir);
    std::sort(vstrImageFilenames.begin(),vstrImageFilenames.end());
}
#endif

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        std::cerr << std::endl << "Usage: descriptor ./test path_to_sequence" << std::endl;
        return 1;
    }

    string descriptor = argv[1];
    cv::Ptr<cv::Feature2D> fdetector;

    if(descriptor == "ORB"){
        std::cout << "Not support ORB yet!!!" << std::endl;
        return -1;
        fdetector = cv::ORB::create();
    }
#ifdef USE_CONTRIB
    else if(descriptor == "SIFT")
        fdetector = cv::xfeatures2d::SIFT::create(500);
#endif

    string dir_name = argv[2];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);


    std::vector<cv::Mat> features;
    for(std::vector<string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
        std:: vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if(img.empty()) throw std::runtime_error("Could not open image" + *i);

        fdetector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
		std::cout << (i - img_file_names.begin()) << " Compute descriptors:" << descriptors.size() << std::endl;

        features.push_back(descriptors);
    }

    KdTreeOptions opt;
    opt._var_threshold = 0.8;
    opt._mean_size = 1000;
    opt._tree_num = 2;
    opt._descriptor = SIFT;
    AKMeans akm(opt);
    akm.TrainTrees(features);

	cv::waitKey(0);

    return 0;
}