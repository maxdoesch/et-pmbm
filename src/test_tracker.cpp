#include "tracker/Tracker.h"
#include "simulator/Simulator.h"
#include "validation/Visualization.h"
#include "validation/Evaluation.h"

int main(int argc, char** argv)
{
    double const time_step = 0.1;

    simulator::Simulator simulator(time_step, 40);
    Eigen::Matrix<double, 5, 1> i_state = Eigen::Matrix<double, 5, 1>::Zero();
    i_state[0] = -6;
    i_state[1] = 0;
    i_state[2] = 0.3;
    i_state[3] = 0;
    i_state[4] = 0;
    simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
    simulator::ExtentModel* e_model = new simulator::UniformEllipse(1, 1, 50);
    simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 1, 30);
    simulator.addTarget(target);

    i_state[0] = 6;
    i_state[1] = 0;
    i_state[2] = -0.3;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::UniformEllipse(1, 1, 50);
    target = new simulator::GenericTarget(k_model, e_model, 1, 30);
    simulator.addTarget(target);

    i_state[0] = -8;
    i_state[1] = -6;
    i_state[2] = 0;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::UniformEllipse(1, 1, 50);
    target = new simulator::GenericTarget(k_model, e_model, 5, 30);
    simulator.addTarget(target);

    i_state[0] = 10;
    i_state[1] = -6;
    i_state[2] = 0;
    i_state[3] = 1;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::UniformEllipse(0.75, 1.25, 70);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    i_state[0] = 10;
    i_state[1] = 5.0;
    i_state[2] = -2;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::UniformEllipse(1.25, 0.75, 50);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    i_state[0] = -10;
    i_state[1] = 7;
    i_state[2] = 1;
    i_state[3] = 0;
    k_model = new simulator::ConstantVelocity(i_state);
    e_model = new simulator::UniformEllipse(1.25, 0.75, 70);
    target = new simulator::GenericTarget(k_model, e_model, 10, 20);
    simulator.addTarget(target);

    validation::Visualization visualization(time_step);
    validation::Evaluation evaluation;

    std::chrono::nanoseconds::rep duration = 0;
    std::chrono::nanoseconds::rep max_duration = 0;
    int i = 0;

    tracker::PMBM pmbm(10);

    while(!simulator.endOfSimulation())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr measurements(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<validation::ValidationModel*> models;

        simulator.step(measurements);

        pmbm.predict(time_step);

        if(measurements->size() > 0)
        {
            auto start = std::chrono::high_resolution_clock::now();

            pmbm.update(measurements);

            auto stop = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            if(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() > max_duration)
                max_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

            i++;
        }

        pmbm.reduce();

        std::vector<validation::ValidationModel*> estimate_models;
        std::vector<validation::ValidationModel*> ground_truth_models;
        simulator.getValidationModels(ground_truth_models);
        pmbm.estimate(estimate_models);

        evaluation.plot(ground_truth_models, estimate_models, simulator.getTime());

        models.insert(models.end(), ground_truth_models.begin(), ground_truth_models.end());
        models.insert(models.end(), estimate_models.begin(), estimate_models.end());

        if(!visualization.draw(measurements, models))
            break;

        visualization.record();

        for(validation::ValidationModel* v_model : models)
        {
            delete v_model;
        }
    }

    std::cout << "Avg. processing time: " << duration / i / 1000000. << "ms; max. processing time: " << max_duration / 1000000. << "ms" << std::endl;

    evaluation.summarize();
    evaluation.draw_plot();
}