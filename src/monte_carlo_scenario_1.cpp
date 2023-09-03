#include "tracker/Tracker.h"
#include "simulator/Simulator.h"
#include "validation/Visualization.h"
#include "validation/Evaluation.h"

int main(int argc, char** argv)
{
    double const time_step = 0.1;
    double const time = 10;
    double const monte_carlo_iterations = 100;

    std::vector<validation::Evaluation> evaluations;

    for(int i = 0; i < monte_carlo_iterations; i++)
    {
        simulator::Simulator simulator(time_step, time);
        
        Eigen::Matrix<double, 2, 1> i_state = Eigen::Matrix<double, 2, 1>::Zero();
        i_state[0] = -15;
        i_state[1] = 4;
        simulator::KinematicModel* k_model = new simulator::Parabola(i_state, 1.25, time);
        simulator::ExtentModel* e_model = new simulator::UniformEllipse(1., 0.5, 40);
        simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 0, time);
        simulator.addTarget(target);
        
        i_state = Eigen::Matrix<double, 2, 1>::Zero();
        i_state[0] = -15;
        i_state[1] = -4;
        k_model = new simulator::Parabola(i_state, 2.25, time);
        e_model = new simulator::UniformEllipse(2, 1.5, 60);
        target = new simulator::GenericTarget(k_model, e_model, 0, time);
        simulator.addTarget(target);

        validation::Visualization visualization(time_step, time);
        validation::Evaluation evaluation;

        tracker::PMBM pmbm(10);

        while(!simulator.endOfSimulation())
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
            std::vector<validation::ValidationModel*> models;

            simulator.step(measurements);

            pmbm.predict(time_step);

            if(measurements->size() > 0)
            {
                pmbm.update(measurements);
            }

            pmbm.reduce();

            std::vector<validation::ValidationModel*> estimate_models;
            std::vector<validation::ValidationModel*> ground_truth_models;
            simulator.getValidationModels(ground_truth_models);
            pmbm.estimate(estimate_models);

            evaluation.plot(ground_truth_models, estimate_models, simulator.getTime());

            models.insert(models.end(), ground_truth_models.begin(), ground_truth_models.end());
            models.insert(models.end(), estimate_models.begin(), estimate_models.end());

            /*
            if(!visualization.draw(measurements, models))
                break;

            visualization.record();
            visualization.plot(ground_truth_models);

            for(validation::ValidationModel* v_model : models)
            {
                delete v_model;
            }*/
        }

        evaluations.push_back(evaluation);
    }

    validation::Evaluation monte_carlo_evaluation(evaluations);
    monte_carlo_evaluation.summarize();
    monte_carlo_evaluation.draw_plot(); 
}