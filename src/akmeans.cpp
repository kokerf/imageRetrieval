#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <list>
#include <set>

#include "akmeans.hpp"
#include "kit.hpp"

bool compare(std::pair<uint32_t, float> i ,std::pair<uint32_t, float> j) { return (i.second <j.second); }

void AKMeans::TrainTrees(std::vector<cv::Mat> &features)
{
    if(opt_.descriptor_ == ORB && features[0].type() != CV_8UC1)
    {
        std::cout << "Wrong type of trainning features!!!" << std::endl;
        return;
    }
    else if (opt_.descriptor_ == SIFT && features[0].type() != CV_32F)
    {
        std::cout << "Wrong type of trainning features!!!" << std::endl;
        return;
    }

    std::vector<cv::Mat> train_features;
    if(features[0].rows != 1){
        TransformFeatures(features, train_features);
    }

    //! initial step: select some festures as means randomly
    SelectMeans(train_features);

    //! loop and make means converge
    double error_last = std::numeric_limits<double>::max();
    for(int t = 0; t < opt_.train_times_; ++t)
    {
        //! creat kdtrees by means
        CreatTrees();

        //! find the most matching mean for each features by descending kdtrees
        std::vector<std::pair<uint32_t, float>> query_results;
        QueryFeatures(train_features, query_results);

        //! count the results by histogram and calculate new means
        double error = CalculateNewMeans(train_features, query_results);
        if((error_last - error)/error < opt_.precision_)
            break;

        error_last = error;
        std::cout << "# loop: " << t << " error:" << error << std::endl 
            << "------------------" <<std::endl;
    }

    GetTreesWords();

    GetWordsWeight(features);
}

void AKMeans::CreatTrees()
{
    trees_.resize(0);
    for(int i = 0; i < opt_.tree_num_; ++i)
    {
        trees_.push_back(KdTree(i));
        trees_.back().CreatTree(means_, opt_);
    }
}

void AKMeans::QueryFeatures(std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results)
{
    results.resize(0);
    for(std::vector<cv::Mat>::iterator ft = total_features.begin(); ft != total_features.end(); ++ft)
    {
        float dist;
        int mean_id = QueryFeature(*ft, dist);
        results.push_back(std::make_pair(mean_id, dist));
    }
}

int AKMeans::QueryFeature(cv::Mat &qurey_feature, float &out_dist)
{
    std::list<Branch> unseen_nodes;
    for(int i = 0; i < trees_.size(); ++i)
    {
        unseen_nodes.push_back(Branch(i, 0, 0.0));
    }

    std::set<uint32_t> means_ids;
    int times = 0;
    while(times++ < opt_.query_times_ && !unseen_nodes.empty())
    {
        unseen_nodes.sort();
        if(unseen_nodes.size() > opt_.queue_size_)
            unseen_nodes.resize(opt_.queue_size_);

        Branch branch = unseen_nodes.front(); unseen_nodes.pop_front();
        KdTree &kdt = trees_[branch.tree_id_];

        NodeId node_id = kdt.Descend(qurey_feature, branch.node_id_, unseen_nodes);
    
        //std::cout << "qrF:" << qurey_feature << std::endl;
        //std::cout << "fdF:" << means[kdt.nodes_[node_id].fid_[0]] << std::endl;

        if(node_id != -1)
            means_ids.insert(kdt.nodes_[node_id].fid_[0]);
    }

    if(!means_ids.empty())
    {
        std::vector<std::pair<uint32_t, float>> results;
        FindKnn(means_, means_ids, qurey_feature, results, 1);

        out_dist = results[0].second;
        return results[0].first;
    }
    else//! it won't happen
    {
        std::cout << " Error!!! can not find the mean which this feature belong to!!!" << std::endl;
        return -1;
    }
}

double AKMeans::CalculateNewMeans(std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results)
{
    assert(total_features.size() == results.size());

    uint32_t means_num = means_.size();
    uint32_t cols = means_[0].cols;

    std::vector<float> dist(means_num, 0.0);
    std::vector<std::vector<uint32_t>> feature_masks;
    feature_masks.resize(opt_.mean_size_);
    for(int i = 0, end = results.size(); i != end; ++i)
    {
        int mean_id = results[i].first;

        feature_masks[mean_id].push_back(i);
        if (total_features[0].type() == CV_8UC1)
            dist[mean_id] += BIN::distance(total_features[i], means_[mean_id]);
        else if (total_features[0].type() == CV_32F)
            dist[mean_id] += results[i].second;
    }

    double error = 0;
    for(int i = 0; i < means_num; ++i)
    {
        cv::Mat &new_mean = means_[i];
        std::vector<uint32_t> &mask = feature_masks[i];

        if(total_features[0].type() == CV_8UC1)
            BIN::meanValue(total_features, new_mean, mask);
        else if(total_features[0].type() == CV_32F)
            FLT::meanValue(total_features, new_mean, mask);

        dist[i] /= mask.size();
        error += dist[i];
    }
        
/*  int hist_height=256;
    int scale = 2;
    int max_val = 100;
    cv::Mat hist_img = cv::Mat::zeros(hist_height, means_num*scale, CV_8UC3);
    for(int i = 0; i< means_num; i++)    
    {    
        float bin_val = hist[i];    
        int intensity = cvRound(bin_val*hist_height/max_val);
 
        cv::rectangle(hist_img, cv::Point(i*scale,hist_height-1),    
            cv::Point((i+1)*scale - 1, hist_height - intensity),    
            CV_RGB(255,255,255));
    }  
  
    cv::imshow("histogram", hist_img);
    cv::waitKey(0);*/

    return error/means_num;
}

void AKMeans::TransformFeatures(const std::vector<cv::Mat> &features, std::vector<cv::Mat> &train_features)
{

    train_features.resize(0);

    train_features.reserve(features.size()*features[0].rows);
    for(int n = 0; n < features.size(); n++)
    {
        for(int i = 0; i < features[n].rows; ++i)
        {
            train_features.push_back(features[n].row(i));
        }
    }

    //features.clear();
    //features.reserve(train_features.size());
    //std::swap(train_features, features);
}

void AKMeans::SelectMeans(std::vector<cv::Mat> &total_features)
{
    uint32_t mean_size = opt_.mean_size_;
    assert(total_features.size() > mean_size);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint32_t> distribution(0, total_features.size()-1);

    std::set<uint32_t> select;
    while(select.size() < mean_size)
    {
        select.insert(distribution(generator));
    }

    means_.resize(0);
    means_.reserve(mean_size);
    for(std::set<uint32_t>::iterator it=select.begin(); it!=select.end(); ++it)
    {
        means_.push_back(total_features.at(*it));
    }
}

void AKMeans::FindKnn(std::vector<cv::Mat> &database, std::set<uint32_t> &mask, cv::Mat &query,
        std::vector<std::pair<uint32_t, float>> &results, uint32_t k)
{
    assert(!database.empty());
    assert(query.rows==1);

    results.resize(0);
    for(std::set<uint32_t>::iterator it=mask.begin(); it!=mask.end(); ++it)
    {
        float dist = 0;
        if(database[0].type() == CV_8UC1)
            dist = BIN::distance(database[*it], query);
        else if(database[0].type() == CV_32F)
            dist = FLT::distance(database[*it], query);

        results.push_back(std::make_pair(*it, dist));
    }

    std::sort(results.begin(), results.end(), compare);

    if(k < results.size())
        results.resize(k);
}

void AKMeans::GetTreesWords()
{
    for(int i = 0; i < opt_.tree_num_; ++i)
    {
        trees_[i].GetWords(means_);
    }
}

void AKMeans::GetWordsWeight(std::vector<cv::Mat> &total_features)
{
    const uint32_t image_num = total_features.size();

    weights_.resize(means_.size(), 0);
    for(uint32_t n = 0; n < image_num; n++)
    {
        std::vector<bool> counted;
        counted.resize(means_.size(), false);

        cv::Mat &features = total_features[n];
        for(uint32_t i = 0; i < features.rows; i++)
        {
            float dist;
            uint32_t mean_id = QueryFeature(features.row(i), dist);

            if(!counted[mean_id])
            {
                weights_[mean_id] ++;
                counted[mean_id] = true;
            }
        }
    }

    for(std::vector<float>::iterator i = weights_.begin(); i != weights_.end(); ++i)
    {
        *i = log((double)image_num / (double)(*i));//! IDF
    }
}

void KdTree::CreatTree(std::vector<cv::Mat> &features, KdtOpt &opt)
{
    opt_ = opt;
    nodes_.resize(0);
    //words_.resize(0);

    assert(!features.empty());

    //! set root node
    nodes_.push_back(Node(0));
    Node &root = nodes_.back();
    root.fid_.resize(features.size());
    for(int i = 0; i < features.size(); ++i)
    {
        root.fid_[i] = i;
    }

    //! split on the root node
    Splitting(0, features);

    std::cout << "Tree: " << this->Id()
    << " nodes:"<< nodes_.size() 
    << " height:" << this->Height() << std::endl;

}

void KdTree::Splitting(NodeId parent_id, std::vector<cv::Mat> &features)
{
    Node &parent = nodes_[parent_id];
    if(parent.fid_.size() == 1)
    {
        //parent.SetVisited();
        //parent.descriptor_ = features[0].clone();
        return;
    }

    //! identify the split dimension for parent node
    GetSplitDimension(parent_id, features);

    if(parent.split_.dim == -1)
        return;

    //! creat child nodes
    Node lChild(nodes_.size());
    Node rChild(nodes_.size() + 1);

    lChild.SetParent(parent);
    rChild.SetParent(parent);
    parent.SetChild(lChild, rChild);
    this->UpdateHeight(parent.Depth() + 1);
    //parent.SetVisited();

    //! assign features to child nodes
    int dim = parent.split_.dim;
    int mean = parent.split_.mean;
    lChild.fid_.resize(0);
    rChild.fid_.resize(0);

    if(features[0].type() == CV_8UC1)
    {
        int col = dim / CHAR_BIT;
        int res = dim % CHAR_BIT;
        uint8_t BIT_CHECK = 0x80 >> res;
        for(std::vector<uint32_t>::iterator i = parent.fid_.begin(), end = parent.fid_.end(); i != end; ++i)
        {

            bool p = features[*i].at<uint8_t>(0, col) & BIT_CHECK;
            if(p)//features[*i].at<uint8_t>(0,dim) < mean)
            {
                rChild.fid_.push_back(*i);
            }
            else
            {
                lChild.fid_.push_back(*i);
            }
        }
    }
    else if(features[0].type() == CV_32F)
    {
        for(std::vector<uint32_t>::iterator i = parent.fid_.begin(), end = parent.fid_.end(); i != end; ++i)
        {
            if(features[*i].at<float>(0,dim) > mean)
            {
                rChild.fid_.push_back(*i);
            }
            else
            {
                lChild.fid_.push_back(*i);
            }
        }
    }

    //! split on child nodes
    if(!lChild.fid_.empty() && !rChild.fid_.empty())
    {
        nodes_.push_back(lChild);
        nodes_.push_back(rChild);

        Splitting(lChild.Id(), features);
        Splitting(rChild.Id(), features);
    }
    else
    //!such as the case: p:2 --> l:2, r:0; it can not happen normally
    {
        Splitting(parent_id, features);
    }
    
}

void KdTree::GetSplitDimension(NodeId node_id, std::vector<cv::Mat> &features)
{
    Node &node = nodes_[node_id];

    node.split_.mean = -1;
    node.split_.dim = -1;

    uint32_t feature_num = node.fid_.size();
    if(feature_num <= 1)
        return;

    //! for ORB
    uint32_t cols = features[0].cols;
    uint32_t ndim = cols;
    if (features[0].type() == CV_8UC1)
        ndim = cols * CHAR_BIT;

    bool *divided_dim;
    divided_dim = new bool[ndim];

    std::fill(divided_dim, divided_dim+ndim, false);
    int pid = node.ParentId();
    //! to disable the dimensions which splited in parent nodes
    while(pid > 0)
    {
        divided_dim[nodes_[pid].split_.dim] = true;
        pid = nodes_[pid].ParentId();
    }

    float *sum, *sum2;
    sum = new float[ndim];
    sum2 = new float[ndim];
    float *var;
    var = new float[ndim];

    std::fill(sum, sum+ndim, 0.0f);
    std::fill(sum2, sum2+ndim, 0.0f);

    float maxvar = -1;
    int split_dim = -1;

    //! statistics on each dimension to get variance
    if(features[0].type() == CV_8UC1)//! too have a check!!!!
    {
        for(int n = 0; n < feature_num; ++n)
        {
            const cv::Mat &d = features[node.fid_[n]];
            const unsigned char *p = d.ptr<unsigned char>();

            for(int i = 0; i < cols; ++i, ++p, sum+=8)
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

        for(int i = 0; i < ndim; ++i)
        {
            sum[i] /= feature_num;
            var[i] = sum[i] - sum[i]*sum[i];

            if(var[i] > maxvar){
                maxvar = var[i];
                split_dim = i;
            }
        }
    }
    else if(features[0].type() == CV_32F)
    {
        for(int n = 0; n < feature_num; ++n)
        {
            const cv::Mat &d = features[node.fid_[n]];
            const float *p = d.ptr<float>();

            for(int i = 0; i < ndim; ++i, ++sum, ++sum2, ++p)
            {
                *sum += (float)*p;
                *sum2 += (float)(*p)* (float)(*p);
            }
            sum-=ndim;
            sum2-=ndim;
        }

        for(int i = 0; i < ndim; ++i)
        {
            sum[i] /= feature_num;
            sum2[i] /= feature_num;
            var[i] = sum2[i] - sum[i]*sum[i];

            if(var[i] > maxvar){
                maxvar = var[i];
                split_dim = i;
            }
        }
    }

/*  for(int i = 0; i < 4; i++)
    {
        std::cout << "F" << i << features[i] << std::endl;
    }*/

    std::vector<uint32_t> highdims;
    //! choose the dimension to split
    if(maxvar >= opt_.min_variance_) //! what if maxvar be zero? *-*?
    {
        float threshold = opt_.var_threshold_ * maxvar;
        for(int i = 0; i < ndim; ++i)
        {
            if(!divided_dim[i] && var[i] >= threshold){
                highdims.push_back(i);
            }
        }

        if(!highdims.empty())
        {
            //! in wondows os, the random_engine is impliemented by rand_s(), so ...
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::uniform_int_distribution<uint32_t> distribution(0, highdims.size()-1);
            split_dim = highdims.at(distribution(generator));
        }
    }

    if(split_dim != -1)
    {
        node.split_.dim = split_dim;
        node.split_.mean = sum[split_dim];
    }
    else
    {   
        std::cout << "============================================" << node.Id() << std::endl;
        std::cout << " Do not find split dimension in node: " << node.Id() << std::endl;
        for (int i = 0; i < node.fid_.size(); i++)
        {
            std::cout << features[node.fid_[i]] << std::endl;
        }

    }

    delete [] divided_dim;
    delete [] sum;
    delete [] sum2;
    delete [] var;
}

NodeId KdTree::Descend(cv::Mat &qurey_feature, NodeId node_id, std::list<Branch> &unseen_nodes)
{
    if(qurey_feature.rows != 1)
    {
        std::cout << "Error!!! The qurey_feature's rows is " << qurey_feature.rows <<std::endl;
        return -1;
    }

    if(node_id >= nodes_.size() || node_id < 0)
    {
        std::cout << " Error when descending at Tree:" 
        << this->Id() << " Node:" << node_id << std::endl;

        return -1;
    }

    if(qurey_feature.type() == CV_8UC1)
    {
        //! loop to find the leaf node
        //std::cout << "dim:" << std::endl;
        while(!nodes_[node_id].IsLeaf())
        {
            Node &refer_node = nodes_[node_id];

            int dim = refer_node.split_.dim;
            float mean = refer_node.split_.mean;

            int col = dim / CHAR_BIT;
            int res = dim % CHAR_BIT;
            bool p = qurey_feature.at<uint8_t>(0, col) & (0x80 >> res);

            int unseen = -1;
            if(p)
            {
                unseen = refer_node.LeftChild();
                node_id = refer_node.RightChild();
            }
            else
            {
                unseen = refer_node.RightChild();
                node_id = refer_node.LeftChild();
            }

            //if(nodes_[unseen].IsValid())
                unseen_nodes.push_back(Branch(this->Id(), unseen, fabs(0)));

            //std::cout << "[" << dim << "," << mean << "] ";
        }
        //std::cout << std::endl;
    }
    else if(qurey_feature.type() == CV_32F)
    {
        while(!nodes_[node_id].IsLeaf())
        {
            Node &refer_node = nodes_[node_id];

            int dim = refer_node.split_.dim;
            float mean = refer_node.split_.mean;

            float dist = qurey_feature.at<float>(0, dim) - mean;

            int unseen = -1;
            if(dist > 0)
            {
                unseen = refer_node.LeftChild();
                node_id = refer_node.RightChild();
            }
            else
            {
                unseen = refer_node.RightChild();
                node_id = refer_node.LeftChild();
            }

            //if (nodes_[unseen].IsValid())//! must be valid
            unseen_nodes.push_back(Branch(this->Id(), unseen, fabs(dist)));
            //else
            //  std::cout << "=========ERROR IN TRAINNING==========" << std::endl;
        }
    }

    return node_id;
}

void KdTree::GetWords(std::vector<cv::Mat> &means)
{
    words_.resize(0);
    for(int n = 0, max = nodes_.size(); n < max; n++)
    {
        if(nodes_[n].IsLeaf())
        {
            Node & node = nodes_[n];
            uint32_t mean_id = node.fid_[0];
            node.SetWordId(words_.size());
            words_.push_back(std::make_pair(&nodes_[n], mean_id));
        }
    }
}