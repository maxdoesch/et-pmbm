#include "simulator/Target.h"
#include "simulator/Simulator.h"
#include "validation/ValidationModel.h"
#include "validation/Visualization.h"
#include "tracker/ExtentModel.h"
#include "tracker/KinematicModel.h"
#include "tracker/Detection.h"
#include "tracker/Bernoulli.h"

#include "gnuplot-iostream.h"

int main(int argc, char** argv)
{   
    double time_step = 0.05;

    double a = 1.0;
    double b = 2.0;
    double p_rate = 100;


    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[2] = 0.2;
    i_state[3] = 0;
    i_state[4] = M_1_PI;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::Ellipse(a, b, p_rate);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 20);
    
    validation::Visualization viz(time_step, 20);
    simulator::Simulator simulator(time_step, 20);
    simulator.addNRandomTargets(4);
    //simulator.addTarget(target);

    tracker::ExtentModel* extentModel = new tracker::GIW<tracker::ConstantVelocity>;
    tracker::Bernoulli bernoulli(extentModel);

    int cnt = 0;

    std::vector<std::pair<double, double>> detection_likelihoods, misdetection_likelihoods;

    while(1)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        if(measurements->points.size() > 0)
        {
            simulator.getValidationModels(models);

            tracker::Cluster detection(measurements);

            tracker::Bernoulli detection_bernoulli(bernoulli);
            double detection_likelihood = detection_bernoulli.detection_likelihood(detection);

            tracker::Bernoulli misdetection_bernoulli(bernoulli);
            double misdetection_likelihood = misdetection_bernoulli.missed_detection_likelihood();
            
            detection_likelihoods.push_back(std::make_pair(simulator.getTime(), detection_likelihood));
            misdetection_likelihoods.push_back(std::make_pair(simulator.getTime(), misdetection_likelihood));

            std::cout << detection_likelihood << "    " << misdetection_likelihood << std::endl;

            if(cnt % 2)
            {
                bernoulli = detection_bernoulli;

                std::cout << "Detected!    Likelihood: " << detection_likelihood << std::endl;
            }
            else
            {
                misdetection_bernoulli.update_missed_detection();
                bernoulli = misdetection_bernoulli;

                std::cout << "Misdetected!    Likelihood: " << misdetection_likelihood << std::endl;
            }
            validation::ValidationModel* estimate = bernoulli.getValidationModel();
            models.push_back(estimate);
        }

        if(!viz.draw(measurements, models))
            break;

        bernoulli.predict(time_step);

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
        models.clear(); 

        cnt++;

        if(simulator.endOfSimulation())
            break;
    }

    // Create a Gnuplot object
    Gnuplot gp;

    // Set the title and labels for the plot
    gp << "set title 'Custom Function Plot'\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";

    // Plot the data
    gp << "plot '-' with lines title 'Function 1', '-' with lines title 'Function 2'\n";
    gp.send1d(detection_likelihoods);
    gp.send1d(misdetection_likelihoods);
}