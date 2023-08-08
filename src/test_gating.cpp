#include "tracker/Preprocessing.h"
#include "tracker/Hypothesis.h"

#include "validation/Visualization.h"

#include "simulator/Simulator.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "gnuplot-iostream.h"


int main(int argc, char** argv)
{
    double const time_step = 0.1;

    simulator::Simulator simulator(time_step, 40);
    simulator.addNRandomTargets(1);
    
    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[0] = -6;
    i_state[1] = 0;
    i_state[2] = 0.3;
    i_state[3] = 0;
    i_state[4] = 0;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::Ellipse(1, 1, 50);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 30);
    simulator.addTarget(target);

    i_state[0] = 6;
    i_state[1] = 0;
    i_state[2] = -0.3;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1, 1, 50);
    target = new simulator::GenericTarget(k_model, e_model, 1, 30);
    simulator.addTarget(target);

    i_state[0] = -8;
    i_state[1] = -6;
    i_state[2] = 0;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1, 1, 50);
    target = new simulator::GenericTarget(k_model, e_model, 5, 30);
    simulator.addTarget(target);

    i_state[0] = 10;
    i_state[1] = -6;
    i_state[2] = 0;
    i_state[3] = 1;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(0.75, 1.25, 70);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    i_state[0] = 10;
    i_state[1] = 5.0;
    i_state[2] = -2;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1.25, 0.75, 50);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    i_state[0] = -10;
    i_state[1] = 7;
    i_state[2] = 1;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1.25, 0.75, 70);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    validation::Visualization visualization(time_step);

    tracker::MultiBernoulliMixture mbm;

    tracker::PPP ppp;
    ppp.predict(time_step);

    std::chrono::nanoseconds::rep duration = 0;
    std::chrono::nanoseconds::rep max_duration = 0;
    int i = 0;

    while(!simulator.endOfSimulation())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        if(mbm.size() == 0)
            mbm.add(tracker::MultiBernoulli());

        mbm.predict(time_step);

        std::cout << "---------------------------------------------" << std::endl;

        if(measurements->size() > 0)
        {
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<tracker::PartitionedParent> parent_partitions;
            tracker::PartitionExtractor partition_extractor(measurements);
            partition_extractor.getPartitionedParents(parent_partitions);

            for(auto const& parent_partition : parent_partitions)
                for(auto const& partition : parent_partition.partitions)
                    partition.getValidationModels(models);

            tracker::MultiBernoulliMixture new_mbm;
            for(auto const& multi_bernoulli : mbm.getMultiBernoullis())
            {
                std::vector<tracker::Group> groups;
                tracker::GroupExtractor group_extractor(parent_partitions, multi_bernoulli.getBernoullis(), ppp);
                group_extractor.extractGroups(groups);
                std::cout << "groups: " << groups.size() << std::endl;

                tracker::Hypotheses hypothesis(multi_bernoulli.getWeight(), groups);
                
                new_mbm.add(hypothesis.getMostLikelyHypotheses(3));
            }
            mbm = new_mbm;

            mbm.normalize();

            auto stop = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            if(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() > max_duration)
                max_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

            i++;
        }

        mbm.prune(-10);
        mbm.capping(5);
        mbm.recycle(0.5);

        mbm.normalize();

        mbm.print();

        std::vector<tracker::Bernoulli> estimate = mbm.estimate(0.89);

        ppp.getValidationModels(models);
        simulator.getValidationModels(models);
        for(auto const& bernoulli : estimate)
            models.push_back(bernoulli.getValidationModel());

        if(!visualization.draw(measurements, models))
            break;

        visualization.record();

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
    }

    std::cout << "Avg. processing time: " << duration / i / 1000000. << "ms; max. processing time: " << max_duration / 1000000. << "ms" << std::endl;
}