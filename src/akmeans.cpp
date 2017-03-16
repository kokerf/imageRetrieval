#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <list>
#include <set>

#include "akmeans.hpp"
#include "kit.hpp"

bool compare(std::pair<uint32_t, float> i ,std::pair<uint32_t, float> j) { return (i.second <j.second); }

void AKMeans::TrainTrees(const std::vector<cv::Mat> &features)
{
    if(_opt._descriptor == ORB && features[0].type() != CV_8UC1)
    {
        std::cout << "Wrong type of trainning features!!!" << std::endl;
        return;
    }
    else if(_opt._descriptor == SIFT && features[0].type() != CV_32F)
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
    for(int t = 0; t < _opt._train_times; ++t)
    {
        //! creat kdtrees by means
        CreatTrees();

        //! find the most matching mean for each features by descending kdtrees
        std::vector<std::pair<uint32_t, float>> query_results;
        QueryFeatures(train_features, query_results);

        //! count the results by histogram and calculate new means
        double error = CalculateNewMeans(train_features, query_results);
        if((error_last - error)/error < _opt._precision)
            break;

        error_last = error;
        std::cout << "# loop: " << t << " error:" << error << std::endl 
            << "------------------" <<std::endl;
    }

    //! use the final means to creat new trees
    CreatTrees();

    GetTreesWords();

    GetWordsWeight(features);
}

void AKMeans::CreatTrees()
{
    _trees.resize(0);
    for(int i = 0; i < _opt._tree_num; ++i)
    {
        _trees.push_back(KdTree(i));
        _trees.back().CreatTree(_means, _opt);
    }
}

void AKMeans::QueryFeatures(const std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results)
{
	uint32_t feature_size = total_features.size();
    results.resize(0);
    for(uint32_t i = 0; i < feature_size; i++)
    {
        float dist;
        MeanId mean_id = QueryFeature(total_features[i], dist);
        results.push_back(std::make_pair(mean_id, dist));
    }
}

MeanId AKMeans::QueryFeature(const cv::Mat &qurey_feature, float &out_dist)
{
    std::list<Branch> unseen_nodes;
    for(int i = 0; i < _trees.size(); ++i)
    {
        unseen_nodes.push_back(Branch(i, 0, 0.0));
    }

    std::set<uint32_t> means_ids;
    int times = 0;
    while(times++ < _opt._query_times && !unseen_nodes.empty())
    {
        unseen_nodes.sort();
        if(unseen_nodes.size() > _opt._queue_size)
            unseen_nodes.resize(_opt._queue_size);

        Branch branch = unseen_nodes.front(); unseen_nodes.pop_front();
        KdTree &kdt = _trees[branch._tree_id];

        NodeId node_id = kdt.Descend(qurey_feature, branch._node_id, unseen_nodes);
    
        //std::cout << "qrF:" << qurey_feature << std::endl;
        //std::cout << "fdF:" << means[kdt._nodes[node_id]._fid[0]] << std::endl;

        if(node_id != -1)
            means_ids.insert(kdt._nodes[node_id]._fid[0]);
    }

    if(!means_ids.empty())
    {
        std::vector<std::pair<uint32_t, float>> results;
        FindKnn(_means, means_ids, qurey_feature, results, 1);

        out_dist = results[0].second;
        return results[0].first;
    }
    else//! it won't happen
    {
        std::cout << " Error!!! can not find the mean which this feature belong to!!!" << std::endl;
        return -1;
    }
}

double AKMeans::CalculateNewMeans(const std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results)
{
    assert(total_features.size() == results.size());

    uint32_t means_num = _means.size();
    uint32_t cols = _means[0].cols;

    std::vector<float> dist(means_num, 0.0);
    std::vector<std::vector<uint32_t>> feature_masks;
    feature_masks.resize(_opt._mean_size);
    for(uint32_t i = 0, end = results.size(); i != end; ++i)
    {
        int mean_id = results[i].first;

        feature_masks[mean_id].push_back(i);
        if (total_features[0].type() == CV_8UC1)
            dist[mean_id] += BIN::distance(total_features[i], _means[mean_id]);
        else if (total_features[0].type() == CV_32F)
            dist[mean_id] += results[i].second;
    }

    double error = 0;
    for(uint32_t i = 0; i < means_num; ++i)
    {
        cv::Mat &new_mean = _means[i];
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
    for(uint32_t n = 0; n < features.size(); n++)
    {
        for(uint32_t i = 0; i < features[n].rows; ++i)
        {
            train_features.push_back(features[n].row(i));
        }
    }

    //features.clear();
    //features.reserve(train_features.size());
    //std::swap(train_features, features);
}

void AKMeans::SelectMeans(const std::vector<cv::Mat> &total_features)
{
    uint32_t mean_size = _opt._mean_size;
    assert(total_features.size() > mean_size);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint32_t> distribution(0, total_features.size()-1);

    std::set<uint32_t> select;
    while(select.size() < mean_size)
    {
        select.insert(distribution(generator));
    }

    _means.resize(0);
    _means.reserve(mean_size);
    for(std::set<uint32_t>::iterator it=select.begin(); it!=select.end(); ++it)
    {
        _means.push_back(total_features.at(*it));
    }
}

void AKMeans::FindKnn(const std::vector<cv::Mat> &database, const std::set<uint32_t> &mask, const cv::Mat &query,
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
    for(int i = 0; i < _opt._tree_num; ++i)
    {
        _words.resize(_means.size());
        _trees[i].GetWords(_means, _words);
    }
}

void AKMeans::GetWordsWeight(const std::vector<cv::Mat> &total_features)
{
    const uint32_t image_num = total_features.size();

    _weights.resize(_means.size(), 0);
    for(uint32_t n = 0; n < image_num; n++)
    {
        std::vector<bool> counted;
        counted.resize(_means.size(), false);

        const cv::Mat &features = total_features[n];
        for(uint32_t i = 0; i < features.rows; i++)
        {
            float dist;
            uint32_t mean_id = QueryFeature(features.row(i), dist);

            if(!counted[mean_id])
            {
                _weights[mean_id] ++;
                counted[mean_id] = true;
            }
        }
    }

    for(std::vector<float>::iterator i = _weights.begin(); i != _weights.end(); ++i)
    {
        *i = log((double)image_num / (double)(*i));//! IDF
    }
}

void AKMeans::Transform(const cv::Mat &features, BowVector &bow_vector)
{
    bow_vector.clear();
    const uint32_t rows = features.rows;
    for(int i = 0; i < rows; ++i)
    {
        float dist;
        WordId word_id = QueryFeature(features.row(i), dist);
        bow_vector.add(word_id, _weights[word_id]);
    }
}

void KdTree::CreatTree(const std::vector<cv::Mat> &features, const KdtOpt &opt)
{
    _opt = opt;
    _nodes.resize(0);
    //words_.resize(0);

    assert(!features.empty());

    //! set root node
    _nodes.push_back(Node(0));
    Node &root = _nodes.back();
    root._fid.resize(features.size());
    for(uint32_t i = 0; i < features.size(); ++i)
    {
        root._fid[i] = i;
    }

    //! split on the root node
    Splitting(0, features);

    std::cout << "Tree: " << this->Id()
    << " nodes:"<< _nodes.size() 
    << " height:" << this->Height() << std::endl;

}

void KdTree::Splitting(NodeId parent_id, const std::vector<cv::Mat> &features)
{
    Node &parent = _nodes[parent_id];
    if(parent._fid.size() == 1)
    {
        //parent.SetVisited();
        //parent._descriptor = features[0].clone();
        return;
    }

    //! identify the split dimension for parent node
    GetSplitDimension(parent_id, features);

    if(parent._split.dim == -1)
        return;

    //! creat child nodes
    Node lChild(_nodes.size());
    Node rChild(_nodes.size() + 1);

    lChild.SetParent(parent);
    rChild.SetParent(parent);
    parent.SetChild(lChild, rChild);
    this->UpdateHeight(parent.Depth() + 1);
    //parent.SetVisited();

    //! assign features to child nodes
    int dim = parent._split.dim;
    int mean = parent._split.mean;
    lChild._fid.resize(0);
    rChild._fid.resize(0);

    if(features[0].type() == CV_8UC1)
    {
        int col = dim / CHAR_BIT;
        int res = dim % CHAR_BIT;
        uint8_t BIT_CHECK = 0x80 >> res;
        for(std::vector<uint32_t>::iterator i = parent._fid.begin(), end = parent._fid.end(); i != end; ++i)
        {

            bool p = features[*i].at<uint8_t>(0, col) & BIT_CHECK;
            if(p)//features[*i].at<uint8_t>(0,dim) < mean)
            {
                rChild._fid.push_back(*i);
            }
            else
            {
                lChild._fid.push_back(*i);
            }
        }
    }
    else if(features[0].type() == CV_32F)
    {
        for(std::vector<uint32_t>::iterator i = parent._fid.begin(), end = parent._fid.end(); i != end; ++i)
        {
            if(features[*i].at<float>(0,dim) > mean)
            {
                rChild._fid.push_back(*i);
            }
            else
            {
                lChild._fid.push_back(*i);
            }
        }
    }

    //! split on child nodes
    if(!lChild._fid.empty() && !rChild._fid.empty())
    {
        _nodes.push_back(lChild);
        _nodes.push_back(rChild);

        Splitting(lChild.Id(), features);
        Splitting(rChild.Id(), features);
    }
    else
    //!such as the case: p:2 --> l:2, r:0; it can not happen normally
    {
        Splitting(parent_id, features);
    }
    
}

void KdTree::GetSplitDimension(NodeId node_id, const std::vector<cv::Mat> &features)
{
    Node &node = _nodes[node_id];

    node._split.mean = -1;
    node._split.dim = -1;

    uint32_t feature_num = node._fid.size();
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
        divided_dim[_nodes[pid]._split.dim] = true;
        pid = _nodes[pid].ParentId();
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
        for(uint32_t n = 0; n < feature_num; ++n)
        {
            const cv::Mat &d = features[node._fid[n]];
            const unsigned char *p = d.ptr<unsigned char>();

            for(uint32_t i = 0; i < cols; ++i, ++p, sum+=8)
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

        for(uint32_t i = 0; i < ndim; ++i)
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
        for(uint32_t n = 0; n < feature_num; ++n)
        {
            const cv::Mat &d = features[node._fid[n]];
            const float *p = d.ptr<float>();

            for(int i = 0; i < ndim; ++i, ++sum, ++sum2, ++p)
            {
                *sum += (float)*p;
                *sum2 += (float)(*p)* (float)(*p);
            }
            sum-=ndim;
            sum2-=ndim;
        }

        for(uint32_t i = 0; i < ndim; ++i)
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
    if(maxvar >= _opt._min_variance) //! what if maxvar be zero? *-*?
    {
        float threshold = _opt._var_threshold * maxvar;
        for(uint32_t i = 0; i < ndim; ++i)
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
        node._split.dim = split_dim;
        node._split.mean = sum[split_dim];
    }
    else
    {   
        std::cout << "============================================" << node.Id() << std::endl;
        std::cout << " Do not find split dimension in node: " << node.Id() << std::endl;
        for (uint32_t i = 0; i < node._fid.size(); i++)
        {
            std::cout << features[node._fid[i]] << std::endl;
        }

    }

    delete [] divided_dim;
    delete [] sum;
    delete [] sum2;
    delete [] var;
}

NodeId KdTree::Descend(const cv::Mat &qurey_feature, NodeId node_id, std::list<Branch> &unseen_nodes)
{
    if(qurey_feature.rows != 1)
    {
        std::cout << "Error!!! The qurey_feature's rows is " << qurey_feature.rows <<std::endl;
        return -1;
    }

    if(node_id >= _nodes.size() || node_id < 0)
    {
        std::cout << " Error when descending at Tree:" 
        << this->Id() << " Node:" << node_id << std::endl;

        return -1;
    }

    if(qurey_feature.type() == CV_8UC1)
    {
        //! loop to find the leaf node
        //std::cout << "dim:" << std::endl;
        while(!_nodes[node_id].IsLeaf())
        {
            Node &refer_node = _nodes[node_id];

            int dim = refer_node._split.dim;
            float mean = refer_node._split.mean;

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

            //if(_nodes[unseen].IsValid())
                unseen_nodes.push_back(Branch(this->Id(), unseen, fabs(0)));

            //std::cout << "[" << dim << "," << mean << "] ";
        }
        //std::cout << std::endl;
    }
    else if(qurey_feature.type() == CV_32F)
    {
        while(!_nodes[node_id].IsLeaf())
        {
            Node &refer_node = _nodes[node_id];

            int dim = refer_node._split.dim;
            float mean = refer_node._split.mean;

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

            //if (_nodes[unseen].IsValid())//! must be valid
            unseen_nodes.push_back(Branch(this->Id(), unseen, fabs(dist)));
            //else
            //  std::cout << "=========ERROR IN TRAINNING==========" << std::endl;
        }
    }

    return node_id;
}

void KdTree::GetWords(const std::vector<cv::Mat> &means, std::vector<std::vector<Node*>> &words)
{
    TreeId tree_id = _id;
    for(uint32_t n = 0, max = _nodes.size(); n < max; n++)
    {
        if(_nodes[n].IsLeaf())
        {
            Node & node = _nodes[n];
            uint32_t mean_id = node._fid[0];
            node.SetWordId(mean_id);//! let word id be mean id
			words[mean_id].push_back(&_nodes[n]);
        }
    }
}