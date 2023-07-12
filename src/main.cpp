#include "simulator/Target.h"
#include "simulator/Simulator.h"
#include "validation/ValidationModel.h"
#include "validation/Visualization.h"
#include "tracker/ExtentModel.h"
#include "tracker/KinematicModel.h"
#include "tracker/Detection.h"

int main(int argc, char** argv)
{   
    double time_step = 0.2;

    double a = 1.0;
    double b = 2.0;
    double p_rate = 100;

    validation::Visualization viz(time_step);
    
    
    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[2] = 0.2;
    i_state[3] = 0;
    i_state[4] = M_1_PI;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::Ellipse(a, b, p_rate);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 20);
    
    simulator::Simulator simulator(time_step, 40);
    //simulator.addNRandomTargets(5);
    simulator.addTarget(target);

    tracker::GIW extentModel(new tracker::ConstantVelocity);
    tracker::RateModel rateModel;

    while(1)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        if(measurements->points.size() > 0)
        {
            simulator.getValidationModels(models);

            tracker::Cluster detection(measurements);
            detection.computeMeanCov();
            extentModel.update(detection);
            rateModel.update(detection);

            validation::GenericValidationModel* estimate = new validation::GenericValidationModel(extentModel.getKinematicValidationModel(), extentModel.getExtentValidationModel(), rateModel.getRateValidationModel());
            models.push_back(estimate);
        }

        if(!viz.draw(measurements, models))
            break;

        extentModel.predict(time_step);
        rateModel.predict();

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
        models.clear(); 

        if(simulator.endOfSimulation())
            break;
    }
}