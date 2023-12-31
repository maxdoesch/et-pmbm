#include "simulator/Simulator.h"
#include "simulator/constants.h"

using namespace simulator;

Simulator::Simulator(double time_step, double e_o_s) : _time_step{time_step}, _end_of_simulation{e_o_s}, _gen{_rd()}
{

}

Simulator::~Simulator()
{
    for(Target* target : _targets)
    {
        delete target;
    }
}

void Simulator::addTarget(Target* target)
{
    _targets.push_back(target);
}

void Simulator::addNRandomTargets(int n)
{
    std::uniform_real_distribution<> uniform(-1, 1);
    std::normal_distribution<> normal;

    for(int i = 0; i < n; i++)
    {
        Eigen::Matrix<double, 5, 1> i_state;
        i_state[0] =  sim_area_x * uniform(_gen);
        i_state[1] =  sim_area_y * uniform(_gen);
        i_state[2] = 0.2 * normal(_gen);
        i_state[3] = 0.2 * normal(_gen);
        i_state[4] = M_PI * uniform(_gen);

        double a = (uniform(_gen) + 2) * 0.5;
        double b = (uniform(_gen) + 2) * 0.5;
        double p_rate = (uniform(_gen) + 1) * 50 + 25;

        double s_o_e = (uniform(_gen) + 1) / 2. * _end_of_simulation / 4.;
        double e_o_e = s_o_e + (uniform(_gen) + 1) / 2. * _end_of_simulation * 3. / 4.;

        simulator::KinematicModel* k_model = new simulator::ConstantVelocity(i_state);
        simulator::ExtentModel* e_model = new simulator::Ellipse(a, b, p_rate);
        simulator::Target* target = new simulator::GenericTarget(k_model, e_model, s_o_e, e_o_e);

        _targets.push_back(target);
    }
}

void Simulator::step(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements)
{
    measurements->clear();

    for(std::vector<Target*>::iterator target = _targets.begin(); target != _targets.end();)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_measurements(new pcl::PointCloud<pcl::PointXYZ>);

        if((*target)->endOfExistence())
        {
            delete *target;
            target = _targets.erase(target);
        }
        else
        {
            (*target)->step(_time, target_measurements);

            *measurements += *target_measurements;

            target++;
        }
    }

    _time += _time_step;
}

void Simulator::getValidationModels(std::vector<validation::ValidationModel*>& models)
{
    for(Target* target : _targets)
    {
        if(target->startOfExistence() && !target->endOfExistence())
            models.push_back(target->getValidationModel());
    }
}

double Simulator::getTime() const
{
    return _time - _time_step;
}

bool Simulator::endOfSimulation() const
{
    return _time > _end_of_simulation || _targets.empty();
}
