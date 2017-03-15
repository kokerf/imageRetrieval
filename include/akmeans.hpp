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
    int tree_num_;
    int mean_size_;
    int max_height_;//! for the random splitting, we can not control the max height of the tree.
    int queue_size_;
    int train_times_;
    int query_times_;
    float min_variance_;
    float var_threshold_;
    float precision_;
    Descriptor descriptor_;

    KdTreeOptions()
    {
        this->tree_num_ = 1;
        this->mean_size_ = 100;
        this->max_height_ = 0;
        this->queue_size_ = 20;
        this->train_times_ = 5;
        this->query_times_ = 5;
        this->min_variance_ = 0.0f;
        this->var_threshold_ = 0.8f;
        this->precision_ = 0.001f;
		this->descriptor_ = ORB;
    };

    KdTreeOptions &operator=(const KdTreeOptions &opt)
    {
        this->tree_num_ = opt.tree_num_;
        this->mean_size_ = opt.mean_size_;
        this->max_height_ = opt.max_height_;
        this->queue_size_ = opt.queue_size_;
        this->train_times_ = opt.train_times_;
        this->query_times_ = opt.query_times_;
        this->min_variance_ = opt.min_variance_;
        this->var_threshold_ = opt.var_threshold_;
        this->precision_ = opt.precision_;
		this->descriptor_ = opt.descriptor_;

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

    TreeId tree_id_;

    NodeId node_id_;

    float distance_;

public:

    Branch(TreeId tree_id = -1, NodeId node_id = -1, float distance = -1){
        tree_id_ = tree_id;
        node_id_ = node_id;
        distance_ = distance;
    }

    bool operator<(Branch& branch) {return this->distance_ < branch.distance_;}
    
};

class Node
{
public:

    Split split_;

    std::vector<uint32_t> fid_;

    //cv::Mat descriptor_;
public:

    Node(NodeId id = -1){
        id_ = id;
        parent_id_ = -1;
        left_child_ = -1;
        right_child_ = -1;
        word_id_ = -1;
        //visited_ = false;
        //isleaf_ = false;

        if(id_ == 0){depth_ = 0;}
        else{ depth_ = -1;}
    };

    ~Node() {fid_.clear();};

    NodeId Id() {return id_;}

    NodeId WordId() {return word_id_;}

    void SetWordId(NodeId id) {word_id_ = id;}

    NodeId ParentId() {return parent_id_;}

    int Depth() {return depth_;}

    void SetDepth(int depth) {depth_ = depth;}

	void SetParent(Node &parent){
        parent_id_ = parent.Id();
        depth_ = parent.Depth() + 1;
    }

    void SetChild(Node &left, Node &right){
        left_child_ = left.Id();
        right_child_ = right.Id();
    }

    NodeId LeftChild() {return left_child_;}

    NodeId RightChild() {return right_child_;}

    //void SetVisited(){visited_ = true;}

    //bool IsVisited(){return visited_;}

    //! can be the case: 2->1,1; 3->(2),1 (2)->1,1;
    //! can not be: 2->(2),0 (2)->1,1
    bool IsLeaf(){
        //if(left_child_==-1 && right_child_==-1)
        if(fid_.size() == 1)
            return true;
        else
            return false;
    }

    //bool IsValid() {return !fid_.empty();}

private:
    NodeId id_;

    NodeId word_id_;

    NodeId parent_id_;

    NodeId left_child_, right_child_;

    int depth_;//! the depth of root node is 0

    //bool visited_;

    //bool isleaf_;
};

class KdTree
{
public:
    KdtOpt opt_;

    std::vector<Node> nodes_;
    std::vector<std::pair<Node*, int32_t>> words_;

    //std::vector<uint32_t> data_;
    
public:
    KdTree(TreeId id) {id_ = id; height_ = 0;}

    KdTree() {id_=-1;}

    ~KdTree() {nodes_.clear(); words_.clear();}

    TreeId Id() {return id_;}

    void CreatTree(std::vector<cv::Mat> &features, KdtOpt &opt);

    void GetSplitDimension(NodeId node_id, std::vector<cv::Mat> &features);

    void Splitting(NodeId parent_id, std::vector<cv::Mat> &features);

    //! For trainning
    NodeId Descend(cv::Mat &qurey_feature, NodeId node_id, std::list<Branch> &unseen_nodes);

    //! For searching
    NodeId Descend(cv::Mat &qurey_feature);

    void UpdateHeight(uint32_t h) {height_ = h;}

    uint32_t Height() {return height_;}

    void GetWords(std::vector<cv::Mat> &means);

private:
    TreeId id_;

    uint32_t height_;
    
};

class AKMeans
{
public:

    KdtOpt opt_;

    std::vector<KdTree> trees_;

    std::vector<cv::Mat> means_;

    std::vector<float> weights_;//! means' weight

    uint32_t N_;//! number of trainning images

public:

    AKMeans(KdtOpt &opt) : opt_(opt) {}

    //AKMeans();

    ~AKMeans() {trees_.clear();}

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