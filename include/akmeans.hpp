#ifndef _AKMEANS_HPP_
#define _AKMEANS_HPP_

#include <stdint.h>
#include <cstdlib>
#include <math.h>
#include <vector>

#include <opencv2/core/core.hpp>


//#define URAND ((double)rand() / ((double)RAND_MAX + 1.))
//get a random number between [X, Y)
//#define RAND(X, Y) (X + URAND * (Y - X))

typedef int32_t NodeId;
typedef int32_t TreeId;

typedef class KdTreeOptions
{
public:
    int tree_num_;
    int mean_size_;
    int max_height_;//! for the random splitting, we can not control the max height of the tree.
	float min_variance;
	float var_threshold;

    KdTreeOptions()
    {
        this->tree_num_ = 1;
        this->mean_size_ = 100;
        this->max_height_ = 0;
        this->min_variance = 0.0f;
        this->var_threshold = 0.8f;
    };

    KdTreeOptions &operator=(const KdTreeOptions &opt)
    {
        this->tree_num_ = opt.tree_num_;
        this->mean_size_ = opt.mean_size_;
        this->max_height_ = opt.max_height_;
        this->var_threshold = opt.var_threshold;

        return *this;
    }

} KdtOpt;

struct Split
{
    float mean;
    int dim;
};

class Node
{
public:

    Split split_;

    std::vector<uint32_t> fid_;

    cv::Mat descriptor_;
public:

    Node(NodeId id = -1){
        id_ = id;
        parent_id_ = -1;
        left_child_ = -1;
        right_child_ = -1;
        //visited_ = false;
        //isleaf_ = false;

        if(id_ == 0){depth_ = 0;}
        else{ depth_ = -1;}
    };

    ~Node(){fid_.clear();};

    NodeId Id(){return id_;}

    NodeId ParentId(){return parent_id_;}

    int Depth(){return depth_;}

    void SetDepth(int depth){depth_ = depth;}

	void SetParent(Node &parent){
        parent_id_ = parent.Id();
        depth_ = parent.Depth() + 1;
    }

    void SetChild(Node &left, Node &right){
        left_child_ = left.Id();
        right_child_ = right.Id();
    }

    NodeId LeftChild(){return left_child_;}

    NodeId RightChild(){return right_child_;}

    //void SetVisited(){visited_ = true;}

    //bool IsVisited(){return visited_;}

    bool IsLeaf(){return IsEmpty();}

    bool IsEmpty(){
        if(left_child_==-1 && right_child_==-1)
            return true;
        else
            return false;
    }

private:
    NodeId id_;

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
    std::vector<uint32_t> leaf_nodes_;

    //std::vector<uint32_t> data_;
    
public:
    KdTree(TreeId id){id_ = id;}

    KdTree(){id_=-1;}

    ~KdTree() { nodes_.clear(); leaf_nodes_.clear(); }

    TreeId Id(){return id_;}

    void CreatTree(std::vector<cv::Mat> &features, KdtOpt &opt);

    void GetSplitDimension(NodeId node_id, std::vector<cv::Mat> &features);

    void Splitting(NodeId parent_id, std::vector<cv::Mat> &features);

    void Descend(std::vector<cv::Mat> &features);

private:
    TreeId id_;

    uint32_t height_;
    
};

class AKMeans
{
public:

    KdtOpt opt_;

    std::vector<KdTree> trees_;

public:

    AKMeans(KdtOpt &opt) : opt_(opt) {}

    //AKMeans();

    ~AKMeans(){trees_.clear();}

    void TrainTrees(std::vector<cv::Mat> &features);

private:

    void TransformFeatures(std::vector<cv::Mat> &features);

    void SelectMeans(std::vector<cv::Mat> &total_features, std::vector<cv::Mat> &select_features);
};



#endif