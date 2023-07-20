
#include "tracker/PoissonPointProcess.h"
#include "tracker/Bernoulli.h"
#include "validation/Visualization.h"
#include "validation/ValidationModel.h"

#include "simulator/Simulator.h"
#include "simulator/Target.h"

int main(int argc, char** argv)
{
    double const time_step = 2;
    double const a = 1.0;
    double const b = 2.0;
    double const p_rate = 100;

    validation::Visualization vizualization(time_step);

    /*
    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[0] = -4;
    i_state[1] = 1;
    i_state[2] = 0.2;
    i_state[3] = 0;
    i_state[4] = -M_1_PI;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::Ellipse(a, b, p_rate);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 20);*/
    
    simulator::Simulator simulator(time_step, 40);
    simulator.addNRandomTargets(1);
    //simulator.addTarget(target);

    pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<validation::ValidationModel*> models;

    tracker::PPP ppp;
    ppp.predict(time_step);
    ppp.getValidationModels(models);

    while(measurements->points.size() == 0 && !simulator.endOfSimulation())
        simulator.step(measurements);

    if(measurements->points.size() == 0)
        return 0;

    tracker::Cluster detection(measurements);
    detection.computeMeanCov();


    tracker::Bernoulli* bernoulli;
    double detection_likelihood = ppp.detection_likelihood(detection, bernoulli);
    models.push_back(bernoulli->getValidationModel());
    
    std::cout << "detection_likelihood: " << detection_likelihood << std::endl;

    for(int i = 0; i < 10; i++)
        ppp.update_missed_detection();
    ppp.getValidationModels(models);

    delete bernoulli;
    detection_likelihood = ppp.detection_likelihood(detection, bernoulli);
    models.push_back(bernoulli->getValidationModel());
    
    std::cout << "detection_likelihood: " << detection_likelihood << std::endl; 

    while(vizualization.draw(measurements, models));
}