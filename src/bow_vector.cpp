#include "bow_vector.hpp"

void BowVector::add(WordId id, WordValue w)
{
    BowVector::iterator mit = this->lower_bound(id);

    if(mit != this->end() && !(this->key_comp()(id, mit->first)))
    {
      mit->second += w;
    }
    else
    {
      this->insert(mit, BowVector::value_type(id, w));
    }
}

void BowVector::Normalize(LNorm norm_type)
{
    double norm = 0.0; 
    BowVector::iterator mit;
    BowVector::iterator end = this->end();

    if(norm_type == L1)
    {
        for(mit = this->begin(); mit != end; ++mit)
            norm += fabs(mit->second);
    }
    else
    {
        for(mit = this->begin(); mit != end; ++mit)
          norm += mit->second * mit->second;
         
        norm = sqrt(norm);
    }

    if(norm > 0.0)
    {
        for(mit = this->begin(); mit != end; ++mit)
          mit->second /= norm;
    }
}