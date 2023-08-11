#include "validation/Evaluation.h"
#include "tracker/utils.h"

#include <gnuplot-iostream.h>

using namespace validation;

Evaluation::Evaluation()
{

}

void Evaluation::plot(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate, double time)
{
    GOSPA gospa(ground_truth, estimate);

    _gospa.push_back(std::make_pair(time, gospa.gospa()));
    _normalized_localization_error.push_back(std::make_pair(time, gospa.normalized_localization_error()));
    _missed_targets.push_back(std::make_pair(time, gospa.missed_targets()));
    _false_targets.push_back(std::make_pair(time, gospa.false_targets()));
}

void Evaluation::draw_plot() const
{
    Gnuplot gp;

    // Set the terminal type to a window
    gp << "set terminal wxt size 1400,1000\n";

    // Set multiplot layout
    gp << "set multiplot layout 2,2 rowsfirst\n";

    // Plot the four datasets in separate plots
    gp << "plot '-' with lines title 'GOSPA'\n";
    gp.send1d(_gospa);

    gp << "plot '-' with lines title 'Normalized Localization Error'\n";
    gp.send1d(_normalized_localization_error);

    gp << "plot '-' with lines title 'Missed Targets'\n";
    gp.send1d(_missed_targets);

    gp << "plot '-' with lines title 'False Targets'\n";
    gp.send1d(_false_targets);

    // Unset multiplot
    gp << "unset multiplot\n";
}

void Evaluation::summarize() const
{
    double _total_gospa = 0;
    for(auto const& gospa : _gospa)
        _total_gospa += gospa.second;

    double _total_nle = 0;
    for(auto const& nle : _normalized_localization_error)
        _total_nle += nle.second;

    double _total_mt = 0;
    for(auto const& mt : _missed_targets)
        _total_mt += mt.second;

    double _total_ft = 0;
    for(auto const& ft : _false_targets)
        _total_ft += ft.second;

    std::cout << "GOSPA: " << _total_gospa << ", NLE: " << _total_nle << ", MT: " << _total_mt << ", FT: " << _total_ft << std::endl;
}   

GOSPA::GOSPA(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate) : 
    _gt_size{ground_truth.size()}, _e_size{estimate.size()}, _cost_matrix(_gt_size, _e_size + _gt_size), _c_pow_p{std::pow(_c, _p)}
{
    _createCostMatrix(ground_truth, estimate);
    _solve();
    //_print();
}

double GOSPA::gospa() const
{
    return pow(_localization_error + _c_pow_p / 2. * (_missed_targets + _false_targets), 1. / _p);
}

double GOSPA::normalized_localization_error() const
{
    return (_localized_targets == 0) ? 0 : _localization_error / _localized_targets;
}

int GOSPA::missed_targets() const
{
    return _missed_targets;
}

int GOSPA::false_targets() const
{
    return _false_targets;
}

void GOSPA::_createCostMatrix(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate)
{
    int j = 0;
    for(auto const& gt : ground_truth)
    {
        int i = 0;
        for(auto const& e : estimate)
        {
            double gaussian_wasserstein_distance = _gaussian_wasserstein_distance(*gt, *e);
            _cost_matrix(j, i) = -std::pow(gaussian_wasserstein_distance, _p);

            i++;
        }

        for(; i < _e_size + j; i++)
            _cost_matrix(j, i) = -std::numeric_limits<double>::infinity();

        _cost_matrix(j, i) = -_c_pow_p;

        for(++i; i < _e_size + _gt_size; i++)
            _cost_matrix(j, i) = -std::numeric_limits<double>::infinity();

        j++;
    }
}

void GOSPA::_solve()
{
    _false_targets = _e_size;

    if(_gt_size != 0)
    {
        _best_assignment = MurtyMiller<double>::getMBestAssignments(_cost_matrix, 1).front();

        for(auto const& assignment : _best_assignment)
        {
            int j = assignment.x;
            int i = assignment.y;

            if(i < _e_size)
            {
                double gaussian_wasserstein_distance = -_cost_matrix(j, i);
                gaussian_wasserstein_distance = (gaussian_wasserstein_distance < _c_pow_p) ? gaussian_wasserstein_distance : _c_pow_p;

                _localization_error += gaussian_wasserstein_distance;

                _localized_targets++;
                _false_targets--;
            }
            else
            {
                _missed_targets++;
            }
        }
    } 
}

double GOSPA::_gaussian_wasserstein_distance(ValidationModel const& model_1, ValidationModel const& model_2) const
{
    Eigen::VectorXd state_1 = model_1.state();
    Eigen::VectorXd state_2 = model_2.state();

    Eigen::MatrixXd extent_1 = model_1.extent();
    Eigen::MatrixXd extent_2 = model_2.extent();

    Eigen::VectorXd state_diff = state_1 - state_2;
    double state_distance = state_diff.transpose() * state_diff;

    Eigen::MatrixXd extent_1_sqrt = matrixSqrt(extent_1);
    Eigen::MatrixXd extent_2_sqrt = matrixSqrt(extent_2);
    
    Eigen::MatrixXd comp_1 = extent_1_sqrt * extent_2 * extent_1_sqrt;
    Eigen::MatrixXd comp_2 = extent_1 + extent_2 - 2 * matrixSqrt(comp_1);
    double extent_distance = comp_2.trace();

    double gaussian_wasserstein_distance = state_distance + extent_distance;
    return gaussian_wasserstein_distance;
}

void GOSPA::_print() const
{
    std::cout << _cost_matrix << "\n";

    
    for (auto const& assignment : _best_assignment)
        std::cout << "(" << assignment.x << ", " << assignment.y << ") ";
    std::cout << "sum = " << MurtyMiller<double>::objectiveFunctionValue(_best_assignment) << std::endl;

}