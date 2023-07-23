#include "tracker/DetectionGroup.h"

#include <limits>
#include <set>

using namespace tracker;

DetectionGroup::DetectionGroup(std::vector<Cluster> const& detections, std::vector<Bernoulli> const& bernoullis, PPP const& ppp) : 
    _d_size{detections.size()}, _b_size{bernoullis.size()}, _costMatrix(detections.size(), bernoullis.size() + detections.size()), _bernoullis{bernoullis}
{   
    _createCostMatrix(detections, bernoullis, ppp);
}

DetectionGroup::~DetectionGroup()
{

}

void DetectionGroup::_createCostMatrix(std::vector<Cluster> const& detections, std::vector<Bernoulli> const& bernoullis, PPP const& ppp)
{   
    int j = 0;
    _bernoulli_matrix.reserve(_d_size);
    for(auto const& detection : detections)
    {
        std::vector<Bernoulli> bernoulli_row;
        bernoulli_row.reserve(_b_size + _d_size);

        int i = 0;
        for(auto const& bernoulli : bernoullis)
        {
            Bernoulli updated_bernoulli(bernoulli);
            _costMatrix(j, i) = updated_bernoulli.detection_likelihood(detection);

            bernoulli_row.push_back(updated_bernoulli);

            i++;
        }

        for(; i < _b_size + j; i++)
            _costMatrix(j, i) = -std::numeric_limits<double>::infinity();

        Bernoulli ppp_bernoulli = ppp.detection_likelihood(detection, _costMatrix(j, _b_size + j) );
        bernoulli_row.push_back(ppp_bernoulli);


        for(++i; i < _b_size + _d_size; i++)
            _costMatrix(j, i) = -std::numeric_limits<double>::infinity();

        _bernoulli_matrix.push_back(bernoulli_row);            

        j++;
    }

    int i = 0;
    for(auto const& bernoulli : bernoullis)
    {
        _costMatrix.col(i) -= bernoulli.missed_detection_likelihood() * Eigen::VectorXd::Ones(_d_size);
        i++;
    }
}



void DetectionGroup::solve(MultiBernoulliMixture& detection_hypotheses)
{
    std::set<int> bernoulli_idx;
    for(int i = 0; i < _b_size; i++)
        bernoulli_idx.insert(i);

    _assignment_hypotheses = MurtyMiller<double>::getMBestAssignments(_costMatrix, _m_assignments);

    for(auto const& assignment_hypothesis : _assignment_hypotheses)
    {
        std::vector<Bernoulli> bernoulli_hypothesis;
        std::set<int> undetected_bernoulli_idx = bernoulli_idx;

        for(auto const& assignment : assignment_hypothesis)
        {
            int j = assignment.x;
            int i = assignment.y;

            if(i < _b_size) 
                undetected_bernoulli_idx.erase(i);
            else
                i = _b_size;

            bernoulli_hypothesis.push_back(Bernoulli((_bernoulli_matrix[j])[i]));
        }

        for(auto const& idx : undetected_bernoulli_idx)
        {
            Bernoulli undetected_bernoulli(_bernoullis[idx]);
            undetected_bernoulli.update_missed_detection();
            bernoulli_hypothesis.push_back(undetected_bernoulli);
        }

        int hypothesis_likelihood = MurtyMiller<double>::objectiveFunctionValue(assignment_hypothesis);
        
        MultiBernoulli multi_bernoulli(bernoulli_hypothesis, hypothesis_likelihood);
        detection_hypotheses.add(multi_bernoulli);
    }
}

void DetectionGroup::print()
{
    std::cout << _costMatrix << "\n";

    for (auto const& assignment_hypothesis : _assignment_hypotheses)
    {
        for (auto const& assignment : assignment_hypothesis)
            std::cout << "(" << assignment.x << ", " << assignment.y << ") ";
        std::cout << "sum = " << MurtyMiller<double>::objectiveFunctionValue(assignment_hypothesis) << std::endl;
    }
}
