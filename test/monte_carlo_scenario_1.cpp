#include "tracker/Tracker.h"
#include "simulator/Simulator.h"
#include "validation/Visualization.h"
#include "validation/Evaluation.h"

int main(int argc, char** argv)
{
    double const time_step = 0.1;
    double const time = 10;
    double const monte_carlo_iterations = 200;

    std::vector<validation::Evaluation> evaluations;

    std::chrono::nanoseconds::rep duration = 0;
    std::chrono::nanoseconds::rep max_duration = 0;

    for(int i = 0; i < monte_carlo_iterations; i++)
    {
        simulator::Simulator simulator(time_step, time);
        
        Eigen::Matrix<double, 2, 1> i_state = Eigen::Matrix<double, 2, 1>::Zero();
        i_state[0] = -12;
        i_state[1] = 4;
        simulator::KinematicModel* k_model = new simulator::Parabola(i_state, 1.1, time);
        simulator::ExtentModel* e_model = new simulator::UniformEllipse(1., 0.5, 50);
        simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 0, time);
        simulator.addTarget(target);
        
        i_state = Eigen::Matrix<double, 2, 1>::Zero();
        i_state[0] = -12;
        i_state[1] = -4;
        k_model = new simulator::Parabola(i_state, 1.1, time);
        e_model = new simulator::UniformEllipse(1., 0.5, 50);
        target = new simulator::GenericTarget(k_model, e_model, 0, time);
        simulator.addTarget(target);

        Eigen::Matrix<double, 5,1> i_state_2 = Eigen::Matrix<double, 5, 1>::Zero();
        i_state_2[0] = 12;
        i_state_2[1] = 7.5;
        k_model = new simulator::ConstantVelocity(i_state_2);
        e_model = new simulator::UniformEllipse(1, 2, 20);
        target = new simulator::GenericTarget(k_model, e_model, 0, time);
        simulator.addTarget(target);

        validation::Visualization visualization(time_step, time);
        validation::Evaluation evaluation;

        auto start = std::chrono::high_resolution_clock::now();

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
            
            visualization.plot(measurements, ground_truth_models);*/

            for(validation::ValidationModel* v_model : models)
            {
                delete v_model;
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        if(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() > max_duration)
            max_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        evaluations.push_back(evaluation);
    }

     std::cout << "Avg. processing time: " << duration / monte_carlo_iterations / 1000000. << "ms; max. processing time: " << max_duration / 1000000. << "ms" << std::endl;

    validation::Evaluation monte_carlo_evaluation(evaluations);
    monte_carlo_evaluation.summarize();
    monte_carlo_evaluation.draw_plot(); 
}