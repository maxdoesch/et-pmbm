#include "tracker/DetectionGroup.h"

#include <limits>
#include <set>

using namespace tracker;

DetectionGroup::DetectionGroup(std::vector<Cluster*> const& detections, std::vector<Bernoulli*> const& bernoullis, PPP const* ppp) : 
    _d_size{detections.size()}, _b_size{bernoullis.size()}, _detections{detections}, _bernoullis{bernoullis}, _ppp{ppp}, _costMatrix(_detections.size(), _bernoullis.size() + _detections.size())
{   
    for(int i = 0; i < _b_size; i++)
        _bernoulli_idx.insert(i);
}

DetectionGroup::~DetectionGroup()
{
    for(auto& bernoulli_row : _bernoulli_matrix)
    {
        for(auto& bernoulli : *bernoulli_row)
            delete bernoulli;

        delete bernoulli_row;
    }
}

void DetectionGroup::createCostMatrix()
{   
    int i = 0;
    for(auto const& bernoulli : _bernoullis)
    {
        _costMatrix.col(i) = -bernoulli->missed_detection_likelihood() * Eigen::VectorXd::Ones(_d_size);
        i++;
    }

    int j = 0;
    for(auto const& detection : _detections)
    {
        std::vector<Bernoulli*>* bernoulli_row = new std::vector<Bernoulli*>;

        i = 0;
        for(auto const& bernoulli : _bernoullis)
        {
            Bernoulli* updated_bernoulli = new Bernoulli(bernoulli);
            _costMatrix(j, i) += updated_bernoulli->detection_likelihood(*detection);

            bernoulli_row->push_back(updated_bernoulli);
            i++;
        }

        for(; i < _b_size + j; i++)
            _costMatrix(j, i) = -std::numeric_limits<double>::infinity();

        Bernoulli* ppp_bernoulli;
        _costMatrix(j, _b_size + j) = _ppp->detection_likelihood(*detection, ppp_bernoulli);
        bernoulli_row->push_back(ppp_bernoulli);

        for(++i; i < _b_size + _d_size; i++)
            _costMatrix(j, i) = -std::numeric_limits<double>::infinity();

        _bernoulli_matrix.push_back(bernoulli_row);            

        j++;
    }
}



void DetectionGroup::solve(std::vector<std::vector<Bernoulli*>*>& bernoulli_hypotheses, std::vector<double>& hypotheses_likelihoods)
{
    _assignment_hypotheses = MurtyMiller<double>::getMBestAssignments(_costMatrix, _m_assignments);

    for(auto const& assignment_hypothesis : _assignment_hypotheses)
    {
        std::vector<Bernoulli*>* bernoulli_hypothesis = new std::vector<Bernoulli*>;
        std::set<int> undetected_bernoulli_idx = _bernoulli_idx;

        for(auto const& assignment : assignment_hypothesis)
        {
            int j = assignment.x;
            int i = assignment.y;

            if(i < _b_size) 
                undetected_bernoulli_idx.erase(i);
            else
                i = _b_size;

            bernoulli_hypothesis->push_back(new Bernoulli((*_bernoulli_matrix[j])[i]));
        }

        for(auto const& idx : undetected_bernoulli_idx)
        {
            Bernoulli* undetected_bernoulli = new Bernoulli(_bernoullis[idx]);
            undetected_bernoulli->update_missed_detection();
            bernoulli_hypothesis->push_back(undetected_bernoulli);
        }

        bernoulli_hypotheses.push_back(bernoulli_hypothesis);

        int likelihood_sum = MurtyMiller<double>::objectiveFunctionValue(assignment_hypothesis);
        hypotheses_likelihoods.push_back(likelihood_sum);
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
