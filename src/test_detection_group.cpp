#include "tracker/PoissonPointProcess.h"
#include "tracker/DetectionGroup.h"

#include "validation/Visualization.h"

#include "simulator/Simulator.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "gnuplot-iostream.h"


void cluster_extractor(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, std::vector<tracker::Cluster*>& detections)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(measurements);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> pcl_euclidean_cluster;
    pcl_euclidean_cluster.setClusterTolerance(2);
    pcl_euclidean_cluster.setMinClusterSize(1);
    pcl_euclidean_cluster.setSearchMethod(tree);
    pcl_euclidean_cluster.setInputCloud(measurements);
    pcl_euclidean_cluster.extract(cluster_indices);

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

        tracker::Cluster* detection = new tracker::Cluster(cluster_measurements);
        detection->computeMeanCov();

        detections.push_back(detection);
    }
}


int main(int argc, char** argv)
{
    double const time_step = 0.1;

    simulator::Simulator simulator(time_step, 40);
    simulator.addNRandomTargets(4);

    validation::Visualization vizualization(time_step);

    std::vector<tracker::Bernoulli*> bernoullis;

    tracker::PPP ppp;
    ppp.predict(time_step);

    std::vector<std::pair<double, double>> hypothesis_likelihood;

    int i = 0;
    while(!simulator.endOfSimulation())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        if(measurements->points.size() > 0)
        {   
            std::vector<tracker::Cluster*> detections;
            cluster_extractor(measurements, detections);

            tracker::DetectionGroup detectionGroup(detections, bernoullis, &ppp);
            detectionGroup.createCostMatrix();

            std::vector<std::vector<tracker::Bernoulli*>*> hypotheses;
            std::vector<double> likelihoods;
            detectionGroup.solve(hypotheses, likelihoods);
            detectionGroup.print();

            for(auto& bernoulli : bernoullis)
                delete bernoulli;

            bernoullis = *hypotheses[0];

            hypothesis_likelihood.push_back(std::make_pair(simulator.getTime(), likelihoods[0]));


            for(auto detection : detections)
            {
                delete detection;
            }

            i++;
        }

        auto bernoulli_it = bernoullis.begin();
        while(bernoulli_it < bernoullis.end())
        {
            (*bernoulli_it)->predict(time_step);
            
            if((*bernoulli_it)->get_pExistence() < 0.1)
            {
                delete *bernoulli_it;
                bernoullis.erase(bernoulli_it);
            }

            bernoulli_it++;
        }

        for(auto& bernoulli : bernoullis)
            models.push_back(bernoulli->getValidationModel());
        ppp.getValidationModels(models);
        simulator.getValidationModels(models);

        if(!vizualization.draw(measurements, models))
            break;

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
    }

    Gnuplot gp;

    // Set the title and labels for the plot
    gp << "set title 'Hypothesis Likelihood'\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";

    // Plot the data
    gp << "plot '-' with lines title 'Function 1'\n";
    gp.send1d(hypothesis_likelihood);

}