#ifndef _BOW_VECTOR_HPP_
#define _BOW_VECTOR_HPP_

#include <stdint.h>
#include <map>

typedef int32_t NodeId;
typedef int32_t WordId;
typedef uint32_t EntryId;
typedef double WordValue;

enum LNorm
{
    L1,
    L2
};

enum ScoreType
{
    L1_Norm,
    L2_Norm,
    // Chi_Square,
    // KL,
    // Bhattacharyya,
    // Dot_Product
};

class BowVector: public std::map<WordId, WordValue>
{
private:

    EntryId _id;

public:

    void add(WordId id, WordValue w);

    void SetEntryId(EntryId id) {_id = id;}

    EntryId EntryId() {return _id;}

    void Normalize(LNorm norm_type);
};

#endif