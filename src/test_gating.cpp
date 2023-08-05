#include "tracker/Preprocessing.h"

#include "validation/Visualization.h"

#include "simulator/Simulator.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "gnuplot-iostream.h"


int main(int argc, char** argv)
{
    double const time_step = 0.1;

    simulator::Simulator simulator(time_step, 40);
    //simulator.addNRandomTargets(2);
    
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

    validation::Visualization visualization(time_step);

    while(!simulator.endOfSimulation())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        if(measurements->size() > 0)
        {
            std::vector<tracker::PartitionedParent> parent_partitions;
            tracker::PartitionExtractor partition_extractor(measurements);
            partition_extractor.getPartitionedParents(parent_partitions);

            for(auto const& parent_partition : parent_partitions)
                for(auto const& partition : parent_partition.partitions)
                    partition.getValidationModels(models);
        }

        if(!visualization.draw(measurements, models))
            break;

        visualization.record();

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
    }
}