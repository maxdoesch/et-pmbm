#include "tracker/Tracker.h"

#include "tracker/Preprocessing.h"
#include "tracker/Detection.h"
#include "tracker/Group.h"
#include "tracker/Hypothesis.h"

using namespace tracker;

PMBM::PMBM(int n_max) : _mbm_results(n_max), _max_number_of_hypotheses{n_max}
{
    _mbm.add(tracker::MultiBernoulli());
}

void PMBM::predict(double ts)
{
    _mbm.predict(ts);
    _ppp.predict(ts);
}

void PMBM::update(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements)
{
    _parent_partitions.clear();
    tracker::PartitionExtractor partition_extractor(measurements);
    partition_extractor.getPartitionedParents(_parent_partitions);

    if(_mbm.size() == 0)
        _mbm.add(tracker::MultiBernoulli());

    boost::thread_group _threads;
    for(int i = 0; i < _mbm.size(); i++)
        _threads.create_thread(boost::bind(&PMBM::_per_multi_bernoulli_update, this, i));

    _threads.join_all();

    MultiBernoulliMixture posterior_mbm;

    for(int i = 0; i < _mbm.size(); i++)
    {
        MultiBernoulliMixture* per_mbm_posterior_mbm;
        while(!_mbm_results.pop(per_mbm_posterior_mbm));

        posterior_mbm.add(*per_mbm_posterior_mbm);
        delete per_mbm_posterior_mbm;
    }

    _mbm = posterior_mbm;
    _ppp.update_missed_detection();
    
    _mbm.normalize();
}

void PMBM::reduce()
{
    _mbm.prune(_mb_pruning_threshold);
    _mbm.capping(_max_number_of_hypotheses);
    _mbm.normalize();
    _mbm.recycle(_bernoulli_recyling_threshold, _ppp);

    _ppp.prune(_ppp_pruning_threshold);
    _ppp.capping(_max_number_of_ppp_components);

    //_mbm.print();
}

void PMBM::estimate(std::vector<validation::ValidationModel*>& estimate_models) const
{
    std::vector<tracker::Bernoulli> estimate = _mbm.estimate(_mbm_estimation_threshold);

    for(auto const& bernoulli : estimate)
        estimate_models.push_back(bernoulli.getValidationModel());
}

void PMBM::_per_multi_bernoulli_update(int idx) 
{
    MultiBernoulli const& multi_bernoulli = _mbm[idx];

    std::vector<tracker::Group> groups;
    tracker::GroupExtractor group_extractor(_parent_partitions, multi_bernoulli.getBernoullis(), _ppp);
    group_extractor.extractGroups(groups);

    tracker::Hypotheses hypotheses(multi_bernoulli.getWeight(), groups);
    MultiBernoulliMixture* mbm_result = new MultiBernoulliMixture(hypotheses.getMostLikelyHypotheses(3));

    while(!_mbm_results.push(mbm_result));
}