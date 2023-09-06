#include "tracker/Tracker.h"
#include "simulator/Simulator.h"
#include "validation/Visualization.h"
#include "validation/Evaluation.h"

int main(int argc, char** argv)
{
    double const time_step = 0.1;
    double const time = 10;
    double const monte_carlo_iterations = 1;

    std::vector<validation::Evaluation> evaluations;

    std::chrono::nanoseconds::rep duration = 0;
    std::chrono::nanoseconds::rep max_duration = 0;

    int nr_x = 4;
    int nr_y = 4;
    double mid_offset = 0.6;
    double ellipse_size = 0.3;
    double measurement_rate = 40;

    Eigen::Matrix<double, 2, 1> pos_0 = Eigen::Matrix<double, 2, 1>::Zero();
    pos_0[0] = mid_offset;
    pos_0[1] = mid_offset;
    Eigen::Matrix<double, 2, 1> pos_1 = Eigen::Matrix<double, 2, 1>::Zero();
    pos_1[0] = mid_offset;
    pos_1[1] = -mid_offset;
    Eigen::Matrix<double, 2, 1> pos_2 = Eigen::Matrix<double, 2, 1>::Zero();
    pos_2[0] = -mid_offset;
    pos_2[1] = -mid_offset;
    Eigen::Matrix<double, 2, 1> pos_3 = Eigen::Matrix<double, 2, 1>::Zero();
    pos_3[0] = -mid_offset;
    pos_3[1] = mid_offset;

    for(int i = 0; i < monte_carlo_iterations; i++)
    {
        simulator::Simulator simulator(time_step, time);
        
        for(int i = 0; i < nr_x; i++)
        {
            double offset_x = 2 * (validation::coordinate_size_x - mid_offset - ellipse_size) /  (nr_x-1) * i - (validation::coordinate_size_x - mid_offset - ellipse_size);
            for(int j = 0; j < nr_y; j++)
            {
                double offset_y = 2 * (validation::coordinate_size_y - mid_offset - ellipse_size) /  (nr_y-1) * j - (validation::coordinate_size_y - mid_offset - ellipse_size);

                Eigen::Vector2d offset = {offset_x, offset_y};

                Eigen::Matrix<double, 5, 1> state_0 = Eigen::Matrix<double, 5, 1>::Zero();
                state_0.block<2,1>(0,0) = pos_0 + offset;
                Eigen::Matrix<double, 5, 1> state_1 = Eigen::Matrix<double, 5, 1>::Zero();
                state_1.block<2,1>(0,0) = pos_1 + offset;
                Eigen::Matrix<double, 5, 1> state_2 = Eigen::Matrix<double, 5, 1>::Zero();
                state_2.block<2,1>(0,0) = pos_2 + offset;
                Eigen::Matrix<double, 5, 1> state_3 = Eigen::Matrix<double, 5, 1>::Zero();
                state_3.block<2,1>(0,0) = pos_3 + offset;

                simulator::KinematicModel* k_model = new simulator::ConstantVelocity(state_0);
                simulator::ExtentModel* e_model = new simulator::UniformEllipse(ellipse_size, ellipse_size, measurement_rate);
                simulator::Target* target = new simulator::GenericTarget(k_model, e_model, 0, time);
                simulator.addTarget(target);

                //k_model = new simulator::ConstantVelocity(state_1);
                //e_model = new simulator::UniformEllipse(ellipse_size, ellipse_size, measurement_rate);
                //target = new simulator::GenericTarget(k_model, e_model, 0, time);
                //simulator.addTarget(target);
                
                //k_model = new simulator::ConstantVelocity(state_2);
                //e_model = new simulator::UniformEllipse(ellipse_size, ellipse_size, measurement_rate);
                //target = new simulator::GenericTarget(k_model, e_model, 0, time);
                //simulator.addTarget(target);

                //k_model = new simulator::ConstantVelocity(state_3);
                //e_model = new simulator::UniformEllipse(ellipse_size, ellipse_size, measurement_rate);
                //target = new simulator::GenericTarget(k_model, e_model, 0, time);
                //simulator.addTarget(target);
            }
        }

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

        evaluations.push_back(evaluation);

        auto stop = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        if(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() > max_duration)
            max_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    }

    std::cout << "Avg. processing time: " << duration / monte_carlo_iterations / 1000000. << "ms; max. processing time: " << max_duration / 1000000. << "ms" << std::endl;

    validation::Evaluation monte_carlo_evaluation(evaluations);
    monte_carlo_evaluation.summarize();
    monte_carlo_evaluation.draw_plot(); 
}