#pragma once

#include "ValidationModel.h"
#include "Miller.h"

namespace validation
{
    class Evaluation
    {
        public:
            Evaluation();
            Evaluation(Evaluation const& Evaluation) = delete;
            Evaluation& operator=(Evaluation const& Evaluation) = delete;

            void plot(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate, double time);
            void draw_plot() const;
            void summarize() const;

        private:
            std::vector<std::pair<double, double>> _gospa;
            std::vector<std::pair<double, double>> _normalized_localization_error;
            std::vector<std::pair<double, int>> _missed_targets;
            std::vector<std::pair<double, int>> _false_targets;
    };


    class GOSPA
    {
        public:
            explicit GOSPA(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate);
            GOSPA(GOSPA const& gospa) = delete;
            GOSPA& operator=(GOSPA const& gospa) = delete;

            double gospa() const;
            double normalized_localization_error() const;
            int missed_targets() const;
            int false_targets() const;

        private:
            void _createCostMatrix(std::vector<ValidationModel*> const& ground_truth, std::vector<ValidationModel*> const& estimate);
            void _solve();
            double _gaussian_wasserstein_distance(ValidationModel const& model_1, ValidationModel const& model_2) const;
            void _print() const;

            std::size_t const _gt_size;
            std::size_t const _e_size;

            MurtyMiller<double>::WeightMatrix _cost_matrix;
            MurtyMiller<double>::Edges _best_assignment;

            double _localization_error = 0;
            double _localized_targets = 0;
            int _missed_targets = 0;
            int _false_targets;

            double const _c = 10;
            int const _p = 1;
            double const _c_pow_p = 0; 
    };
}