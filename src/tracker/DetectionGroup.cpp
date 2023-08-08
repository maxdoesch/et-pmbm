#include "tracker/DetectionGroup.h"

#include "tracker/constants.h"

#include <limits>

using namespace tracker;

DetectionGroup::DetectionGroup(std::vector<Cluster> const& detections, std::vector<Bernoulli> const& prior_hypothesis, PPP const& ppp) : 
    _d_size{detections.size()}, _b_size{prior_hypothesis.size()}, _cost_matrix(detections.size(), prior_hypothesis.size() + detections.size()), _prior_hypothesis{prior_hypothesis}
{   
    _createCostMatrix(detections, ppp);
}

DetectionGroup::~DetectionGroup()
{

}

void DetectionGroup::_createCostMatrix(std::vector<Cluster> const& detections, PPP const& ppp)
{   
    int j = 0;
    _updated_bernoulli_matrix.reserve(_d_size);
    for(auto const& detection : detections)
    {
        std::vector<Bernoulli> bernoulli_row;
        bernoulli_row.reserve(_b_size + _d_size);

        int i = 0;
        for(auto const& prior_hypothesis_bernoulli : _prior_hypothesis)
        {
            Bernoulli updated_bernoulli(prior_hypothesis_bernoulli);
            _cost_matrix(j, i) = updated_bernoulli.detection_likelihood(detection);

            bernoulli_row.push_back(updated_bernoulli);

            i++;
        }

        for(; i < _b_size + j; i++)
            _cost_matrix(j, i) = -std::numeric_limits<double>::infinity();

        Bernoulli ppp_bernoulli = ppp.detection_likelihood(detection, _cost_matrix(j, _b_size + j) );
        bernoulli_row.push_back(ppp_bernoulli);


        for(++i; i < _b_size + _d_size; i++)
            _cost_matrix(j, i) = -std::numeric_limits<double>::infinity();

        _updated_bernoulli_matrix.push_back(bernoulli_row);            

        j++;
    }

    int i = 0;
    for(auto const& prior_hypothesis_bernoulli : _prior_hypothesis)
    {
        _cost_matrix.col(i) -= prior_hypothesis_bernoulli.missed_detection_likelihood() * Eigen::VectorXd::Ones(_d_size);
        i++;
    }
}



void DetectionGroup::solve(MultiBernoulliMixture& detection_hypotheses)
{
    std::set<int> bernoulli_idx;
    for(int i = 0; i < _b_size; i++)
        bernoulli_idx.insert(i);

    if(_d_size > 0)
        _assignment_hypotheses = MurtyMiller<double>::getMBestAssignments(_cost_matrix, _m_assignments);
    else
    {
        std::vector<Bernoulli> undetected_bernoullis;
        double hypothesis_likelihood = 0;

        for(auto const& prior_bernoulli : _prior_hypothesis)
        {
            Bernoulli undetected_bernoulli(prior_bernoulli);
            hypothesis_likelihood += undetected_bernoulli.missed_detection_likelihood();
            undetected_bernoulli.update_missed_detection();
            undetected_bernoullis.push_back(undetected_bernoulli);
        }

        MultiBernoulli multi_bernoulli(undetected_bernoullis, hypothesis_likelihood);
        detection_hypotheses.add(multi_bernoulli);
    }

    for(auto const& assignment_hypothesis : _assignment_hypotheses)
    {
        std::vector<Bernoulli> assignment_hypothesis_bernoulli;
        std::set<int> undetected_bernoulli_idx = bernoulli_idx;

        for(auto const& assignment : assignment_hypothesis)
        {
            int j = assignment.x;
            int i = assignment.y;

            if(i < _b_size) 
                undetected_bernoulli_idx.erase(i);
            else
                i = _b_size;

            assignment_hypothesis_bernoulli.push_back(Bernoulli((_updated_bernoulli_matrix[j])[i]));
        }

        for(auto const& idx : undetected_bernoulli_idx)
        {
            Bernoulli undetected_bernoulli(_prior_hypothesis[idx]);
            undetected_bernoulli.update_missed_detection();
            assignment_hypothesis_bernoulli.push_back(undetected_bernoulli);
        }

        double hypothesis_likelihood = MurtyMiller<double>::objectiveFunctionValue(assignment_hypothesis);
        //hypothesis_likelihood = (hypothesis_likelihood < min_likelihood) ? min_likelihood : hypothesis_likelihood;

        MultiBernoulli multi_bernoulli(assignment_hypothesis_bernoulli, hypothesis_likelihood);
        detection_hypotheses.add(multi_bernoulli);
    }
}

void DetectionGroup::print()
{
    std::cout << _cost_matrix << "\n";

    for (auto const& assignment_hypothesis : _assignment_hypotheses)
    {
        for (auto const& assignment : assignment_hypothesis)
            std::cout << "(" << assignment.x << ", " << assignment.y << ") ";
        std::cout << "sum = " << MurtyMiller<double>::objectiveFunctionValue(assignment_hypothesis) << std::endl;
    }
}
