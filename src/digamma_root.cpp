#include "tracker/utils.h"

#include <vector>
#include <cmath>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"
#include <boost/math/distributions/gamma.hpp>

#include <chrono>

int main()
{
    double weight_1 = 0.8;
    double alpha_1 = 5;
    double beta_1 = 6;
    boost::math::gamma_distribution<double> gamma_dist_1(alpha_1, 1 / beta_1);

    double weight_2 = 0.2;
    double alpha_2 = 5;
    double beta_2 = 7;
    boost::math::gamma_distribution<double> gamma_dist_2(alpha_2, 1 / beta_2);

    double weight[] = {weight_1, weight_2};
    double alpha[] = {alpha_1, alpha_2};
    double beta[] = {beta_1, beta_2};

    double alpha_3, beta_3;

    auto start = std::chrono::high_resolution_clock::now();
    merge_gamma(alpha_3, beta_3, weight, alpha, beta, 2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    boost::math::gamma_distribution<double> gamma_dist_3(alpha_3, 1 / beta_3);

    std::cout << "alpha: " << alpha_3 << " beta: " << beta_3 << " time: " << duration << std::endl;


    // Create a Gnuplot object
    Gnuplot gp;

    // Set the title and labels for the plot
    gp << "set title 'Custom Function Plot'\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";

    // Create a vector to store the data points
    std::vector<std::pair<double, double>> data_1, data_2, data_3;

    // Generate function values and store them in the data vector
    for (double x = 0; x <= 10.0; x += 0.01) {
        double y = boost::math::pdf(gamma_dist_1, x);
        data_1.push_back(std::make_pair(x, y));

        y = boost::math::pdf(gamma_dist_2, x);
        data_2.push_back(std::make_pair(x, y));

        y = boost::math::pdf(gamma_dist_3, x);
        data_3.push_back(std::make_pair(x, y));
    }

    // Plot the data
    gp << "plot '-' with lines title 'Function 1', '-' with lines title 'Function 2', '-' with lines title 'Function 3'\n";
    gp.send1d(data_1);
    gp.send1d(data_2);
    gp.send1d(data_3);

    return 0;
}