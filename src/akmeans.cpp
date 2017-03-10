#include <iostream>
#include <random>
#include <vector>
#include <list>
#include <set>

#include "akmeans.hpp"
#include "distance.hpp"

bool compare(std::pair<uint32_t, float> i ,std::pair<uint32_t, float> j) { return (i.second <j.second); }

void AKMeans::TrainTrees(std::vector<cv::Mat> &features)
{
    if(features[0].rows != 1)
        TransformFeatures(features);

    //! initial step: select some festures as means randomly
    std::vector<cv::Mat> means;
    SelectMeans(features, means);

    //! loop and make means converge
    for(int t = 0; t < opt_.train_times_; ++t)
    {
        //! creat kdtrees by means
        CreatTrees(means);

        //! find the most matching mean for each features by descending kdtrees
        std::vector<std::pair<uint32_t, float>> query_results;
        query_results.resize(features.size());
        QueryFeatures(features, means, query_results);

        //! count the results by histogram and calculate new means
        means.clear();
        float error = CalculateNewMeans(features, means, query_results);
    }
    
}

void AKMeans::CreatTrees(std::vector<cv::Mat> &means)
{
    trees_.resize(0);
    for(int i = 0; i < opt_.tree_num_; ++i)
    {
        trees_.push_back(KdTree(i));
        trees_.back().CreatTree(means, opt_);
    }
}

void AKMeans::QueryFeatures(std::vector<cv::Mat> &total_features, std::vector<cv::Mat> &means,
        std::vector<std::pair<uint32_t, float>> &results)
{
    for(std::vector<cv::Mat>::iterator ft = total_features.begin(); ft != total_features.end(); ++ft)
    {
		float dist;
        int mean_id = QueryFeature(means, *ft, dist);
    }
}

int AKMeans::QueryFeature(std::vector<cv::Mat> &means, cv::Mat &qurey_feature, float &out_dist)
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
    
        if(kdt.nodes_[node_id].IsValid())
            means_ids.insert(kdt.nodes_[node_id].fid_[0]);
    }

	if(!means_ids.empty())
    {
        std::vector<std::pair<uint32_t, float>> results;
        FindKnn(means, means_ids, qurey_feature, results, 1);

        out_dist = results[0].second;
        return results[0].first;
    }
    else//! it won't happen
    {
        std::cout << " Error!!! can not find the mean which this feature belong to!!!" << std::endl;
        return -1;
    }
}

float AKMeans::CalculateNewMeans(std::vector<cv::Mat> &total_features, std::vector<cv::Mat> &means,
        std::vector<std::pair<uint32_t, float>> &results)
{
    //! TODO!!!!
	return 0;
}

void AKMeans::TransformFeatures(std::vector<cv::Mat> &features)
{
    std::vector<cv::Mat> features_temp;
    features_temp.resize(0);

    for(std::vector<cv::Mat>::iterator fin = features.begin(); fin != features.end(); ++fin)
    {
        for(int i = 0; i < fin->rows; ++i)
        {
            features_temp.push_back(fin->row(i).clone());
        }
    }

    features.clear();
    //features.reserve(features_temp.size());
    std::swap(features_temp, features);
}

void AKMeans::SelectMeans(std::vector<cv::Mat> &total_features, std::vector<cv::Mat> &select_features)
{
    uint32_t mean_size = opt_.mean_size_;
    //select_features.clear();

    std::default_random_engine generator;
    std::uniform_int_distribution<uint32_t> distribution(0, total_features.size()-1);

    std::set<uint32_t> select;
    while(select.size() < mean_size)
    {
        select.insert(distribution(generator));
    }

    select_features.resize(0);
    for(std::set<uint32_t>::iterator it=select.begin(); it!=select.end(); ++it)
    {
        select_features.push_back(total_features.at(*it));
    }
}

void AKMeans::FindKnn(std::vector<cv::Mat> &database, std::set<uint32_t> &mask, cv::Mat &query,
        std::vector<std::pair<uint32_t, float>> &results, uint32_t k)
{
    if(query.rows != 1 || database.empty())//! whether is necessary?
    {
        std::cout << "Error input of FindKnn!!!" <<std::endl;
        return;
    }

    results.resize(0);
    for(std::set<uint32_t>::iterator it=mask.begin(); it!=mask.end(); ++it)
    {
		Distance DIST;
        float dist = DIST.distance(database[*it], query);

		results.push_back(std::make_pair(*it, dist));
    }

    std::sort(results.begin(), results.end(), compare);

	if(k < results.size())
		results.resize(k);

}

void KdTree::CreatTree(std::vector<cv::Mat> &features, KdtOpt &opt)
{
    opt_ = opt;
    nodes_.resize(0);
    leaf_nodes_.resize(0);

    if(features.empty())
        std::cerr << "No enough features for splitting!!!" << std::endl;

    //! 2^n < mean size < 2^(n+1)
    //uint32_t max_node_num = 2 * opt_.mean_size_ - 1;

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
            << " nodes:"<< nodes_.size() << std::endl;

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
    Node rChild(nodes_.size()+1);

    lChild.SetParent(parent);
    rChild.SetParent(parent);
    parent.SetChild(lChild, rChild);
    //parent.SetVisited();

    //! assign features to child nodes
    int dim = parent.split_.dim;
    int mean = parent.split_.mean;
    lChild.fid_.resize(0);
    rChild.fid_.resize(0);
    for(std::vector<uint32_t>::iterator i = parent.fid_.begin(), end = parent.fid_.end(); i != end; ++i)
    {
        //! for CV_8UC1
        if(features[*i].at<uint8_t>(0,dim) < mean)
        {
            lChild.fid_.push_back(*i);
        }
        else
        {
            rChild.fid_.push_back(*i);
        }
    }

    nodes_.push_back(lChild);
    nodes_.push_back(rChild);

    //! split on child nodes
    if(!lChild.fid_.empty())
    {
        Splitting(lChild.Id(), features);
    }
    
    if(!rChild.fid_.empty())
    {
        Splitting(rChild.Id(), features);
    }
    
}

void KdTree::GetSplitDimension(NodeId node_id, std::vector<cv::Mat> &features)
{
    Node &node = nodes_[node_id];

    node.split_.mean = -1;
    node.split_.dim = -1;

    uint32_t feature_num = node.fid_.size();
    if(feature_num == 1)
        return;

    uint32_t ndim = features[0].cols;

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

    std::fill(sum, sum+ndim, 0.0f);
    std::fill(sum2, sum2+ndim, 0.0f);

    //! statistics on each dimension to get variance
    if(features[0].type() == CV_8UC1)//! too have a check!!!!
    {
        for(int n = 0; n < feature_num; ++n)
        {
            const cv::Mat &d = features[node.fid_[n]];
            const unsigned char *p = d.ptr<unsigned char>();

            for(int i = 0; i < ndim; ++i, ++sum, ++sum2, ++p)
            {
                *sum += (float)*p;
                *sum2 += (float)(*p)* (float)(*p);
            }
            sum-=ndim;
            sum2-=ndim;
        }
    }
    else if(features[0].type() == CV_32F)
    {
        //! TODO!!!
    }

    float *var;
    var = new float[ndim];
    float maxvar = -1;
    for(int i = 0; i < ndim; ++i)
    {
        sum[i] /= feature_num;
        sum2[i] /= feature_num;
        var[i] = sum2[i] - sum[i]*sum[i];

        if(var[i] > maxvar){
            maxvar = var[i];
        }
    }

    std::vector<uint32_t> highdims;
    int split_dim = -1; 
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
            std::default_random_engine generator;
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
        std::cout << " Do not find split dimension in node: " << node.Id() << std::endl;
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
        while(!nodes_[node_id].IsLeaf())
        {
            Node &qurey_node = nodes_[node_id];

            int dim = qurey_node.split_.dim;
            float mean = qurey_node.split_.mean;

            float dist = qurey_feature.at<uint8_t>(0, dim) - mean;

            int unseen = -1;
            if(dist < 0)
            {
                unseen = qurey_node.RightChild();
                node_id = qurey_node.LeftChild();
            }
            else
            {
                unseen = qurey_node.LeftChild();
                node_id = qurey_node.RightChild();
            }

            if(nodes_[unseen].IsValid())
                unseen_nodes.push_back(Branch(this->Id(), unseen, fabs(dist)));

        }
    }
    else if(qurey_feature.type() == CV_32F)
    {
        //! TODO!!!
        return -1;
    }

    return node_id;
}