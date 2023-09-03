#include "tracker/PoissonPointProcess.h"
#include "tracker/DetectionGroup.h"
#include "tracker/Group.h"
#include "tracker/Hypothesis.h"

#include "validation/Visualization.h"

#include "simulator/Simulator.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "gnuplot-iostream.h"


void cluster_extractor(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, std::vector<tracker::Cluster>& detections, double radius)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(measurements);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> pcl_euclidean_cluster;
    pcl_euclidean_cluster.setClusterTolerance(radius);
    pcl_euclidean_cluster.setMinClusterSize(3);
    pcl_euclidean_cluster.setSearchMethod(tree);
    pcl_euclidean_cluster.setInputCloud(measurements);
    pcl_euclidean_cluster.extract(cluster_indices);

    detections.reserve(cluster_indices.size());
    for (const pcl::PointIndices & cluster : cluster_indices) 
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_measurements(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto & point_idx : cluster.indices) 
        {
            cluster_measurements->points.push_back(measurements->points[point_idx]);
        }

        cluster_measurements->width = cluster_measurements->points.size();
        cluster_measurements->height = 1;
        cluster_measurements->is_dense = false;

        tracker::Cluster detection(cluster_measurements);

        detections.push_back(detection);
    }
}

void partition_extractor(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, std::vector<tracker::Partition>& partitions)
{
    int n_detections = 0;

    for(int i = 15; i > 1; i--)
    {
        tracker::Partition new_partition(measurements, 0.15 * i);

        if(n_detections < new_partition.detections.size())
        {
            n_detections = new_partition.detections.size();

            partitions.push_back(new_partition);
        }
    }
}


int main(int argc, char** argv)
{
    double const time_step = 0.1;

    simulator::Simulator simulator(time_step, 40);
    //simulator.addNRandomTargets(2);
    
    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[0] = -2;
    i_state[1] = -2;
    i_state[2] = 0.8;
    i_state[3] = 0;
    i_state[4] = 0;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::Ellipse(1, 1, 50);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 20);
    simulator.addTarget(target);

    i_state[0] = 2;
    i_state[1] = 1.5;
    i_state[2] = -0.3;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1., 1, 50);
    target = new simulator::GenericTarget(k_model, e_model, 1, 20);
    simulator.addTarget(target);

    i_state[0] = 8;
    i_state[1] = -1.5;
    i_state[2] = 0;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::Ellipse(1, 1.5, 20);
    target = new simulator::GenericTarget(k_model, e_model, 4, 20);
    simulator.addTarget(target);

    validation::Visualization visualization(time_step, 40);

    tracker::MultiBernoulliMixture mbm;

    tracker::PPP ppp;
    ppp.predict(time_step);

    std::chrono::nanoseconds::rep duration = 0;
    int i = 0;

    while(!simulator.endOfSimulation())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        auto start = std::chrono::high_resolution_clock::now();

        if(mbm.size() == 0)
            mbm.add(tracker::MultiBernoulli());

        mbm.predict(time_step);

        if(measurements->size() > 0)
        {
            std::vector<tracker::Partition> partitions;
            partition_extractor(measurements, partitions);
            
            /*for(auto const& partition : partitions)
                partition.getValidationModels(models);*/

            tracker::MultiBernoulliMixture new_mbm;
            for(auto const& multi_bernoulli : mbm.getMultiBernoullis())
            {
                std::vector<tracker::Group> groups;
                tracker::Group group(partitions, multi_bernoulli.getBernoullis(), ppp);
                groups.push_back(group);

                tracker::Hypotheses hypothesis(multi_bernoulli.getWeight(), groups);
                
                new_mbm.add(hypothesis.getMostLikelyHypotheses(3));
            }
            mbm = new_mbm;

            mbm.normalize();


            auto stop = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            i++;
        }

        std::cout << "-------------------------------------------------------------------" << std::endl; 
        //mbm.print();

        mbm.prune(-10);
        mbm.capping(5);
        mbm.prune_bernoulli(0.5);

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

    std::cout << "Avg. processing time: " << duration / i / 1000000. << "ms" << std::endl;
}