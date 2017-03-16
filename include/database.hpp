#ifndef _DATA_BASE_HPP_
#define _DATA_BASE_HPP_

#include <vector>
#include <list>
#include <utility>

#include "akmeans.hpp"
#include "ScoringObject.h"
#include "QueryResults.h"


using DBoW3::QueryResults;
using DBoW3::Result;

typedef class DataBaseOptions
{
public:

    bool _inv_file;//! InvertedFile
    bool _dir_file;//! DirectFile
    LNorm _norm;
    ScoreType _score_type;

public:
    DataBaseOptions()
    {
        _inv_file = true;
        _dir_file = false;
        _norm = L1;
        _score_type = L1_Norm;
    }

    ~DataBaseOptions(){}

} DbOpt;

typedef std::pair<EntryId, double> iFPair;

typedef class InverseFile: 
	public std::list<iFPair>
{
    
} InvFile;

template <typename T>
class DataBase
{
public:

    DbOpt _opt;

    T _voc; //! vocabulary

    std::vector<BowVector> _bow_vectors;

    std::vector<InvFile> _inv_files;

public:

    DataBase(DbOpt &opt, T &voc) : _opt(opt), _voc(voc)
    {
        if(_opt._inv_file)
        {
            _inv_files.resize(_voc._words.size());
        }
    }

    ~DataBase(){}

    void AddImages(const std::vector<cv::Mat> &features)
    {
        const uint32_t image_num = features.size();
        for(int i = 0; i < image_num; ++i)
        {
            BowVector bow_vector;
            EntryId entry_id = AddImage(features[i], bow_vector);
        }
    }

    EntryId AddImage(const cv::Mat &features, BowVector &bow_vector)
    {
        _voc.Transform(features, bow_vector);

        EntryId entry_id = _bow_vectors.size();
        bow_vector.SetEntryId(entry_id);
        _bow_vectors.push_back(bow_vector);

        for(BowVector::iterator mit = bow_vector.begin(); mit != bow_vector.end(); ++mit)
        {
            const WordId &word_id = mit->first;
            const float &word_weight = mit->second;
            _inv_files[word_id].push_back(std::make_pair(entry_id, word_weight));
        }

        return entry_id;
    }

    void query(const  cv::Mat &features, QueryResults &ret, int max_results, int max_id)
    {
		BowVector vf;
		AddImage(features, vf);
        query(vf, ret, max_results, max_id);
    }

    void query(const BowVector &query, QueryResults &ret, int max_results, int max_id)
    {
        switch(_opt._score_type)
        {
            case L1_Norm: queryL1(query, ret, max_results, max_id);break;
            case L2_Norm: queryL1(query, ret, max_results, max_id);break;
        }
    }

    void queryL1(const BowVector &vec, QueryResults &ret, int max_results, int max_id)
    {
      BowVector::const_iterator vit;

      std::map<EntryId, double> pairs;
      std::map<EntryId, double>::iterator pit;

      for(vit = vec.begin(); vit != vec.end(); ++vit)
      {
        const WordId word_id = vit->first;
        const WordValue& qvalue = vit->second;

        const InvFile& iv_file = _inv_files[word_id];

        // InvFile are sorted in ascending entry_id order
        for(auto rit = iv_file.begin(); rit != iv_file.end(); ++rit)
        {
          const EntryId entry_id = rit->first;
          const WordValue& dvalue = rit->second;

          if((int)entry_id < max_id || max_id == -1)
          {
            double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);

            pit = pairs.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
              pit->second += value;
            }
            else
            {
              pairs.insert(pit,
                std::map<EntryId, double>::value_type(entry_id, value));
            }
          }

        } // for each inverted row
      } // for each query word

      // move to vector
      ret.reserve(pairs.size());
      for(pit = pairs.begin(); pit != pairs.end(); ++pit)
      {
        ret.push_back(Result(pit->first, pit->second));
      }

      // resulting "scores" are now in [-2 best .. 0 worst]

      // sort vector in ascending order of score
      std::sort(ret.begin(), ret.end());
      // (ret is inverted now --the lower the better--)

      // cut vector
      if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

      // complete and scale score to [0 worst .. 1 best]
      // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
      //        for all i | v_i != 0 and w_i != 0
      // (Nister, 2006)
      // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
      QueryResults::iterator qit;
      for(qit = ret.begin(); qit != ret.end(); qit++)
        qit->Score = -qit->Score/2.0;
    }

    // --------------------------------------------------------------------------


    void queryL2(const BowVector &vec, DBoW3::QueryResults &ret, int max_results, int max_id) const
    {
      BowVector::const_iterator vit;

      std::map<EntryId, double> pairs;
      std::map<EntryId, double>::iterator pit;

      //map<EntryId, int> counters;
      //map<EntryId, int>::iterator cit;

      for(vit = vec.begin(); vit != vec.end(); ++vit)
      {
        const WordId word_id = vit->first;
        const WordValue& qvalue = vit->second;

        const InvFile& iv_file = _inv_files[word_id];

        // InvFile are sorted in ascending entry_id order

        for(auto rit = iv_file.begin(); rit != iv_file.end(); ++rit)
        {
          const EntryId entry_id = rit->first;
          const WordValue& dvalue = rit->second;

          if((int)entry_id < max_id || max_id == -1)
          {
            double value = - qvalue * dvalue; // minus sign for sorting trick

            pit = pairs.lower_bound(entry_id);
            //cit = counters.lower_bound(entry_id);
            if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
            {
              pit->second += value;
              //cit->second += 1;
            }
            else
            {
              pairs.insert(pit,
                std::map<EntryId, double>::value_type(entry_id, value));

              //counters.insert(cit,
              //  map<EntryId, int>::value_type(entry_id, 1));
            }
          }

        } // for each inverted row
      } // for each query word

      // move to vector
      ret.reserve(pairs.size());
      //cit = counters.begin();
      for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
      {
        ret.push_back(Result(pit->first, pit->second));// / cit->second));
      }

      // resulting "scores" are now in [-1 best .. 0 worst]

      // sort vector in ascending order of score
      std::sort(ret.begin(), ret.end());
      // (ret is inverted now --the lower the better--)

      // cut vector
      if(max_results > 0 && (int)ret.size() > max_results)
        ret.resize(max_results);

      // complete and scale score to [0 worst .. 1 best]
      // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i)
        //      for all i | v_i != 0 and w_i != 0 )
        // (Nister, 2006)
        QueryResults::iterator qit;
      for(qit = ret.begin(); qit != ret.end(); qit++)
      {
        if(qit->Score <= -1.0) // rounding error
          qit->Score = 1.0;
        else
          qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
          // the + sign is ok, it is due to - sign in
          // value = - qvalue * dvalue
      }

    }

    // --------------------------------------------------------------------------
    
};

#endif