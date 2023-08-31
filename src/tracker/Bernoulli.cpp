#include "tracker/Bernoulli.h"
#include "tracker/PoissonPointProcess.h"
#include "tracker/utils.h"
#include "tracker/constants.h"

#include <queue>

using namespace tracker;

Bernoulli::Bernoulli(ExtentModel* e_model) : _e_model{e_model}
{

}

Bernoulli::Bernoulli(Bernoulli const& bernoulli) :
    _e_model{bernoulli._e_model->copy()}, _r_model(bernoulli._r_model), _p_existence{bernoulli._p_existence}
{
}

Bernoulli::Bernoulli(double p_existence, ExtentModel* e_model, RateModel const& r_model) : _e_model{e_model}, _r_model{r_model}, _p_existence{p_existence}
{

}

Bernoulli::~Bernoulli()
{
    delete _e_model;
}

void Bernoulli::predict(double ts)
{
    _p_existence = p_survival * _p_existence;

    _e_model->predict(ts);
    _r_model.predict();
}

double Bernoulli::missed_detection_likelihood() const
{  
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();

    double qd = 1. - p_detection + p_detection * pow(beta / (beta + 1), alpha);

    double likelihood = std::log(1 - _p_existence + _p_existence * qd);

    return likelihood;
}

double Bernoulli::detection_likelihood(Cluster const& detection)
{
    double update_likelihood = _e_model->update(detection);
    update_likelihood += _r_model.update(detection);
    double bernoulli_likelihood = std::log(_p_existence) + std::log(p_detection) + update_likelihood;

    _p_existence = 1;

    return bernoulli_likelihood;
}

void Bernoulli::update_missed_detection()
{
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();
    double qd = 1. - p_detection + p_detection * pow(beta / (beta + 1), alpha);

    _p_existence = _p_existence * qd / (1 - _p_existence + _p_existence * qd);

    double alpha_c[] = {alpha, alpha};
    double beta_c[] = {beta, beta + 1};
    double weight_c[] = {1 / qd * (1 - p_detection), 1 / qd * p_detection * pow(beta / (beta + 1), alpha)};

    double alpha_m, beta_m;
    merge_gamma(alpha_m, beta_m, weight_c, alpha_c, beta_c, 2);

    _r_model = RateModel(alpha_m, beta_m);
}

double Bernoulli::get_pExistence() const
{
    return _p_existence; 
}

double Bernoulli::squared_distance(Cluster const& detection) const
{
    return _e_model->squared_distance(detection);
}

validation::ValidationModel* Bernoulli::getValidationModel() const
{
    return new validation::GenericValidationModel(_e_model->getKinematicValidationModel(), _e_model->getExtentValidationModel(), _r_model.getRateValidationModel(), CV_RGB(255, 0, 0));
}

void Bernoulli::operator=(Bernoulli const& bernoulli)
{
    delete _e_model;
    _e_model = bernoulli._e_model->copy();
    _r_model = bernoulli._r_model;
    _p_existence = bernoulli._p_existence;
}

void Bernoulli::print() const
{
    std::cout << "      B: " << std::endl;
    std::cout << "      pE: " << _p_existence << std::endl;
    std::cout << "      ---" << std::endl;
}

MultiBernoulli::MultiBernoulli(std::vector<Bernoulli> const& bernoullis, double weight) 
    : _bernoullis{bernoullis}, _weight{weight}
{
    
}

MultiBernoulli::MultiBernoulli(MultiBernoulli const& multi_bernoulli) 
    : _bernoullis{multi_bernoulli._bernoullis}, _weight{multi_bernoulli._weight}
{

}

void MultiBernoulli::predict(double ts)
{
    for(auto& bernoulli : _bernoullis)
    {
        bernoulli.predict(ts);
    }
}

void MultiBernoulli::prune(double threshold)
{
    std::vector<tracker::Bernoulli>::iterator bernoulli_iterator;
    for(bernoulli_iterator = _bernoullis.begin(); bernoulli_iterator < _bernoullis.end();)
    {
        if((*bernoulli_iterator).get_pExistence() < threshold)
        {
            bernoulli_iterator = _bernoullis.erase(bernoulli_iterator);
        }
        else
            bernoulli_iterator++;
    }
}

void MultiBernoulli::recycle(double threshold, PPP& ppp)
{
    std::vector<tracker::Bernoulli>::iterator bernoulli_iterator;
    for(bernoulli_iterator = _bernoullis.begin(); bernoulli_iterator < _bernoullis.end();)
    {
        if((*bernoulli_iterator).get_pExistence() < threshold)
        {
            ppp.add_component(PoissonComponent(_weight, *bernoulli_iterator));
            bernoulli_iterator = _bernoullis.erase(bernoulli_iterator);
        }
        else
            bernoulli_iterator++;
    }
}

void MultiBernoulli::join(MultiBernoulli const& bernoullis)
{
    _weight += bernoullis._weight;

    for(auto const& bernoulli : bernoullis._bernoullis)
        _bernoullis.push_back(bernoulli);
}

std::vector<Bernoulli> const& MultiBernoulli::getBernoullis() const
{
    return _bernoullis;
}

double MultiBernoulli::getWeight() const
{
    return _weight;
}

void MultiBernoulli::setWeight(double weight)
{
    _weight = weight;
}

int MultiBernoulli::size() const
{
    return _bernoullis.size();
}

void MultiBernoulli::getValidationModels(std::vector<validation::ValidationModel*>& models) const
{
    for(auto const& bernoulli : _bernoullis)
    {
        models.push_back(bernoulli.getValidationModel());
    }
}

void MultiBernoulli::print() const
{
    std::cout << "  MB: " << _bernoullis.size() << " components" << std::endl;
    std::cout << "  weight: " << _weight << std::endl;

    for(auto const& bernoulli : _bernoullis)
    {
        bernoulli.print();
    }

    std::cout << "  -------" << std::endl;
}

void MultiBernoulli::operator=(MultiBernoulli const& multi_bernoulli)
{
    _bernoullis = multi_bernoulli._bernoullis;
    _weight = multi_bernoulli._weight;
}

bool MultiBernoulli::operator>(MultiBernoulli const& other) const
{
    return _weight > other._weight;
}

MultiBernoulliMixture::MultiBernoulliMixture()
{

}

MultiBernoulliMixture::MultiBernoulliMixture(MultiBernoulliMixture const& multi_bernoulli_mixture) : _multi_bernoullis{multi_bernoulli_mixture._multi_bernoullis}
{

}

MultiBernoulliMixture::MultiBernoulliMixture(double prev_weight, std::vector<std::vector<MultiBernoulli>> group_mbs, int M)
{
    struct ComparePair
    {
        bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first; 
        }
    };

    for(auto& multi_bernoulli : group_mbs)
        std::sort(multi_bernoulli.begin(), multi_bernoulli.end(), std::greater<MultiBernoulli>());

    int m = 0;

    int max_mb_size = 0;
    for(auto const& group_mb : group_mbs)
        max_mb_size = (max_mb_size < group_mb.size()) ? group_mb.size() : max_mb_size;

    MultiBernoulli out_of_range_mb;
    for(int j = 0; j < max_mb_size && m < M; j++)
    {
        std::vector<std::pair<double, int>> diff_queue;
        for(int i = 0; i < group_mbs.size(); i++)
        {
            if(j < group_mbs[i].size() - 1)
            {
                diff_queue.push_back(std::make_pair(group_mbs[i][j]._weight - group_mbs[i][j+1]._weight, i));
            }
            else
            {
                out_of_range_mb.join(group_mbs[i][j]);
                group_mbs.erase(group_mbs.begin() + i);
                i--;
            }
        }
        std::sort(diff_queue.begin(), diff_queue.end(), ComparePair());

        if(group_mbs.empty())
        {
            _multi_bernoullis.push_back(out_of_range_mb);
        }

        for(int i = 0; i < std::pow(2, group_mbs.size()) - 1 && m < M; i++)
        {
            MultiBernoulli multi_bernoulli;
            multi_bernoulli.join(out_of_range_mb);

            bool do_not_insert = false;
            for(int a = 0; a < group_mbs.size(); a++)
            {
                int idx = diff_queue[a].second;
                int value = diff_queue[a].first;
                if(i & (1 << a))
                {
                    if(value < 0)
                    {
                        do_not_insert = true;
                        break;
                    }
                    
                    multi_bernoulli.join(group_mbs[idx][j+1]);
                }
                else
                    multi_bernoulli.join(group_mbs[idx][j]);
            }

            if(!do_not_insert)
            {
                _multi_bernoullis.push_back(multi_bernoulli);
                m++;
            }
        }
    }

    for(auto& multi_bernoulli : _multi_bernoullis)
        multi_bernoulli._weight += prev_weight;
}

void MultiBernoulliMixture::predict(double ts)
{
    
    for(auto& multi_bernoulli : _multi_bernoullis)
        multi_bernoulli.predict(ts);
}

void MultiBernoulliMixture::prune(double threshold)
{
    std::vector<tracker::MultiBernoulli>::iterator multi_bernoulli_iterator;
    for(multi_bernoulli_iterator = _multi_bernoullis.begin(); multi_bernoulli_iterator < _multi_bernoullis.end();)
    {
        if((*multi_bernoulli_iterator).getWeight() < threshold)
        {
            multi_bernoulli_iterator = _multi_bernoullis.erase(multi_bernoulli_iterator);
        }
        else
            multi_bernoulli_iterator++;
    }
}

void MultiBernoulliMixture::capping(int N)
{
    std::priority_queue<MultiBernoulli, std::vector<MultiBernoulli>, std::greater<MultiBernoulli>> minHeap;

    for(auto const& multi_bernoulli : _multi_bernoullis)
    {
        if(minHeap.size() < N)
            minHeap.push(multi_bernoulli);
        else if(multi_bernoulli.getWeight() > minHeap.top().getWeight())
        {
            minHeap.pop();
            minHeap.push(multi_bernoulli);
        }
    }

    _multi_bernoullis.clear();
    _multi_bernoullis.reserve(minHeap.size());

    while(!minHeap.empty())
    {
        _multi_bernoullis.push_back(minHeap.top());
        minHeap.pop();
    }
}

void MultiBernoulliMixture::prune_bernoulli(double threshold)
{
    for(auto& multi_bernoulli : _multi_bernoullis)
        multi_bernoulli.prune(threshold);
}

void MultiBernoulliMixture::recycle(double threshold, PPP& ppp)
{
    for(auto& multi_bernoulli : _multi_bernoullis)
        multi_bernoulli.recycle(threshold, ppp);
}

std::vector<Bernoulli> MultiBernoulliMixture::estimate(double threshold) const
{
    std::vector<Bernoulli> estimate;

    if(!_multi_bernoullis.empty())
    {
        MultiBernoulli const* mb_max = &_multi_bernoullis.front();
        for(auto const& multi_bernoulli : _multi_bernoullis)
        {
            if(multi_bernoulli.getWeight() > mb_max->getWeight())
                mb_max = &multi_bernoulli;
        }

        for(auto const& bernoulli : mb_max->getBernoullis())
        {
            if(bernoulli.get_pExistence() > threshold)
                estimate.push_back(bernoulli);
        }
    }

    return estimate;
}

void MultiBernoulliMixture::print() const
{
    std::cout << "MBM: " << _multi_bernoullis.size() << " components" << std::endl;
    for(auto const& multi_bernoulli : _multi_bernoullis)
    {
        multi_bernoulli.print();
    }

    std::cout << "---------" << std::endl;
}

void MultiBernoulliMixture::merge(double prev_weight, std::vector<MultiBernoulli> const& multi_bernoullis)
{
    int i = 0;
    for(auto const& multi_bernoulli : multi_bernoullis)
    {
        if(i >= _multi_bernoullis.size())
        {
            double weight = prev_weight + multi_bernoulli.getWeight();
            MultiBernoulli new_multi_bernoulli(multi_bernoulli.getBernoullis(), weight);
            _multi_bernoullis.push_back(new_multi_bernoulli);
        }
        else
            _multi_bernoullis[i].join(multi_bernoulli);

        i++;
    }
}

void MultiBernoulliMixture::normalize()
{
    std::vector<double> l_weights;
    l_weights.reserve(_multi_bernoullis.size());

    for(auto const& multi_bernoulli : _multi_bernoullis)
    {
        l_weights.push_back(multi_bernoulli.getWeight());
    }
    
    double l_weight_sum = sum_log_weights(l_weights);

    for(auto& multi_bernoulli : _multi_bernoullis)
    {
        double normal_weight = multi_bernoulli.getWeight() - l_weight_sum;

        multi_bernoulli.setWeight(normal_weight);
    }
}

void MultiBernoulliMixture::add(MultiBernoulli const& multi_bernoulli)
{
    _multi_bernoullis.push_back(multi_bernoulli);
}

void MultiBernoulliMixture::add(MultiBernoulliMixture const& multi_bernoulli_mixture)
{
    for(auto const& multi_bernoulli : multi_bernoulli_mixture._multi_bernoullis)
        _multi_bernoullis.push_back(multi_bernoulli);
}

void MultiBernoulliMixture::clear()
{
    _multi_bernoullis.clear();
}

std::vector<MultiBernoulli> MultiBernoulliMixture::selectMostLikely(int x) const
{
    int n = (_multi_bernoullis.size() < x) ? _multi_bernoullis.size() : x;  

    std::priority_queue<MultiBernoulli, std::vector<MultiBernoulli>, std::greater<MultiBernoulli>> minHeap;

    for(auto const& multi_bernoulli : _multi_bernoullis)
    {
        if(minHeap.size() < n)
            minHeap.push(multi_bernoulli);
        else if(multi_bernoulli.getWeight() > minHeap.top().getWeight())
        {
            minHeap.pop();
            minHeap.push(multi_bernoulli);
        }
    }

    std::vector<MultiBernoulli> mostLikelyMixture;

    while(!minHeap.empty())
    {
        mostLikelyMixture.push_back(minHeap.top());
        minHeap.pop();
    }
    /*
    for(int i = n; i < x; i++)
    {
        mostLikelyMixture.push_back(mostLikelyMixture.back());
    }*/

    return mostLikelyMixture;
}

std::vector<MultiBernoulli> const& MultiBernoulliMixture::getMultiBernoullis()
{
    return _multi_bernoullis;
}

MultiBernoulli& MultiBernoulliMixture::operator[](int idx)
{
    return _multi_bernoullis[idx];
}

void MultiBernoulliMixture::operator=(MultiBernoulliMixture const& multi_bernoulli_mixture)
{
    _multi_bernoullis = multi_bernoulli_mixture._multi_bernoullis;
}

int MultiBernoulliMixture::size() const
{
    return _multi_bernoullis.size();
}