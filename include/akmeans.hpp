#ifndef _AKMEANS_HPP_
#define _AKMEANS_HPP_

#include <stdint.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <list>
#include <set>
#include <utility> //! std::pair, std::make_pair

#include <opencv2/opencv.hpp>

//#define URAND ((double)rand() / ((double)RAND_MAX + 1.))
//get a random number between [X, Y)
//#define RAND(X, Y) (X + URAND * (Y - X))

typedef int32_t NodeId;
typedef int32_t TreeId;

typedef enum Descriptor
{
    ORB = 0,
	SIFT,
}Descriptor;

typedef class KdTreeOptions
{
public:
    int _tree_num;
    int _mean_size;
    int _max_height;//! for the random splitting, we can not control the max height of the tree.
    int _queue_size;
    int _train_times;
    int _query_times;
    float _min_variance;
    float _var_threshold;
    float _precision;
    Descriptor _descriptor;

    KdTreeOptions()
    {
        this->_tree_num = 1;
        this->_mean_size = 100;
        this->_max_height = 0;
        this->_queue_size = 20;
        this->_train_times = 5;
        this->_query_times = 5;
        this->_min_variance = 0.0f;
        this->_var_threshold = 0.8f;
        this->_precision = 0.001f;
		this->_descriptor = SIFT;
    };

    KdTreeOptions &operator=(const KdTreeOptions &opt)
    {
        this->_tree_num = opt._tree_num;
        this->_mean_size = opt._mean_size;
        this->_max_height = opt._max_height;
        this->_queue_size = opt._queue_size;
        this->_train_times = opt._train_times;
        this->_query_times = opt._query_times;
        this->_min_variance = opt._min_variance;
        this->_var_threshold = opt._var_threshold;
        this->_precision = opt._precision;
		this->_descriptor = opt._descriptor;

        return *this;
    }

} KdtOpt;

struct Split
{
    float mean;
    int dim;
};

//! for searching unseen nodes
class Branch
{
public:

    TreeId _tree_id;

    NodeId _node_id;

    float _distance;

public:

    Branch(TreeId tree_id = -1, NodeId node_id = -1, float distance = -1){
        _tree_id = tree_id;
        _node_id = node_id;
        _distance = distance;
    }

    bool operator<(Branch& branch) {return this->_distance < branch._distance;}
    
};

class Node
{
public:

    Split _split;

    std::vector<uint32_t> _fid;

    //cv::Mat descriptor_;
public:

    Node(NodeId id = -1){
        _id = id;
		_parent_id = -1;
        _left_child = -1;
        _right_child = -1;
        _word_id = -1;
        //visited_ = false;
        //isleaf_ = false;

        if(_id == 0){_depth = 0;}
        else{ _depth = -1;}
    };

    ~Node() {_fid.clear();};

    NodeId Id() {return _id;}

    NodeId WordId() {return _word_id;}

    void SetWordId(NodeId id) {_word_id = id;}

    NodeId ParentId() {return _parent_id;}

    int Depth() {return _depth;}

    void SetDepth(int depth) {_depth = depth;}

	void SetParent(Node &parent){
        _parent_id = parent.Id();
        _depth = parent.Depth() + 1;
    }

    void SetChild(Node &left, Node &right){
        _left_child = left.Id();
        _right_child = right.Id();
    }

    NodeId LeftChild() {return _left_child;}

    NodeId RightChild() {return _right_child;}

    //void SetVisited(){_visited_ = true;}

    //bool IsVisited(){return _visited_;}

    //! can be the case: 2->1,1; 3->(2),1 (2)->1,1;
    //! can not be: 2->(2),0 (2)->1,1
    bool IsLeaf(){
        //if(_left_child_==-1 && _right_child_==-1)
        if(_fid.size() == 1)
            return true;
        else
            return false;
    }

    //bool IsValid() {return !_fid_.empty();}

private:
    NodeId _id;

    NodeId _word_id;

    NodeId _parent_id;

    NodeId _left_child, _right_child;

    int _depth;//! the depth of root node is 0

    //bool _visited;

    //bool _isleaf;
};

class KdTree
{
public:
    KdtOpt _opt;

    std::vector<Node> _nodes;
    std::vector<std::pair<Node*, int32_t>> _words;

    //std::vector<uint32_t> data_;
    
public:
    KdTree(TreeId id) {_id = id; _height = 0;}

    KdTree() {_id=-1;}

    ~KdTree() {_nodes.clear(); _words.clear();}

    TreeId Id() {return _id;}

    void CreatTree(std::vector<cv::Mat> &features, KdtOpt &opt);

    void GetSplitDimension(NodeId node_id, std::vector<cv::Mat> &features);

    void Splitting(NodeId parent_id, std::vector<cv::Mat> &features);

    //! For trainning
    NodeId Descend(cv::Mat &qurey_feature, NodeId node_id, std::list<Branch> &unseen_nodes);

    //! For searching
    NodeId Descend(cv::Mat &qurey_feature);

    void UpdateHeight(uint32_t h) {_height = h;}

    uint32_t Height() {return _height;}

    void GetWords(std::vector<cv::Mat> &means);

private:
    TreeId _id;

    uint32_t _height;
    
};

class AKMeans
{
public:

    KdtOpt _opt;

    std::vector<KdTree> _trees;

    std::vector<cv::Mat> _means;

    std::vector<float> _weights;//! means' weight

public:

    AKMeans(KdtOpt &opt) : _opt(opt) {}

    //AKMeans();

    ~AKMeans() {_trees.clear();}

    void TrainTrees(std::vector<cv::Mat> &features);

    void GetBowVector();

private:

    void CreatTrees();

    void TransformFeatures(const std::vector<cv::Mat> &features, std::vector<cv::Mat> &train_features);

    void SelectMeans(std::vector<cv::Mat> &total_features);

    void QueryFeatures(std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results);
    
    int QueryFeature(cv::Mat &qurey_feature, float &out_dist);

    double CalculateNewMeans(std::vector<cv::Mat> &total_features, std::vector<std::pair<uint32_t, float>> &results);

    void FindKnn(std::vector<cv::Mat> &database, std::set<uint32_t> &mask, cv::Mat &query,
        std::vector<std::pair<uint32_t, float>> &results, uint32_t k);

    void GetTreesWords();

    void GetWordsWeight(std::vector<cv::Mat> &total_features);
};



#endif