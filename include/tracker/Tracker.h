#pragma once

#include <boost/thread.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/lockfree/queue.hpp>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "tracker/Bernoulli.h"
#include "tracker/PoissonPointProcess.h"

namespace tracker
{
    class PMBM
    {
        public:
            explicit PMBM(int n_max);
            PMBM(PMBM const& pmbm) = delete;
            PMBM& operator=(PMBM const& pmbm) = delete;

            void predict(double ts);
            void update(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements);
            void reduce();
            void estimate(std::vector<validation::ValidationModel*>& estimate_models) const;

        private:
            void _per_multi_bernoulli_update(int idx);

            PPP _ppp;
            MultiBernoulliMixture _mbm;
            std::vector<PartitionedParent> _parent_partitions;

            boost::lockfree::queue<MultiBernoulliMixture*> _mbm_results;
            
            int const _max_number_of_hypotheses;
            double const _mb_pruning_threshold = -40;
            double const _bernoulli_recyling_threshold = 0.1;
            double const _mbm_estimation_threshold = 0.5;

            double const _ppp_pruning_threshold = 1e-7;
            int const _max_number_of_ppp_components = 5;
    };
}