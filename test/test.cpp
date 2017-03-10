#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "akmeans.hpp"

using std::string;

#include <io.h>
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

int main(int argc, char const *argv[])
{
	if(argc != 2)
	{
	    std::cerr << std::endl << "Usage: ./test path_to_sequence" << std::endl;
	    return 1;
	}

	string dir_name = argv[1];
	std::vector<string> img_file_names;
	loadImages(dir_name, img_file_names);

	cv::Ptr<cv::Feature2D> fdetector;
	fdetector = cv::ORB::create();

	std::vector<cv::Mat> features;
    for(std::vector<string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
		std:: vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
		if (img.empty())throw std::runtime_error("Could not open image" + *i);

		fdetector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
		features.push_back(descriptors);
    }

	KdTreeOptions opt;
	opt.var_threshold = 0.8;
	opt.mean_size_ = 100;
	opt.tree_num_ = 2;
	AKMeans akm(opt);
	akm.TrainTrees(features);

	return 0;
}