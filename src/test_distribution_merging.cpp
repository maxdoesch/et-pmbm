#include "tracker/utils.h"

#include <vector>
#include <cmath>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>

#include <chrono>

void merge_gamma()
{
    double weight_1 = 0.3;
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
    gp << "set title 'Merge Gamma'\n";
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
}

void merge_gaussian()
{
    double weight_1 = 0.5;
    Eigen::Vector4d mean_1 = Eigen::Vector4d::Ones() * 4;
    Eigen::Matrix4d cov_1 = Eigen::Matrix4d::Identity() * 4;
    boost::math::normal_distribution<double> normal_dist_1(mean_1[0], cov_1(0,0));

    double weight_2 = 0.5;
    Eigen::Vector4d mean_2 = Eigen::Vector4d::Ones() * -4;
    Eigen::Matrix4d cov_2 = Eigen::Matrix4d::Identity() * 4;
    boost::math::normal_distribution<double> normal_dist_2(mean_2[0], cov_1(0,0));

    double weight[] = {weight_1, weight_2};
    Eigen::Vector4d mean[] = {mean_1, mean_2};
    Eigen::Matrix4d cov[] = {cov_1, cov_2};

    Eigen::Vector4d mean_3;
    Eigen::Matrix4d cov_3;

    auto start = std::chrono::high_resolution_clock::now();
    merge_gaussian(mean_3, cov_3, weight, mean, cov, 2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    boost::math::normal_distribution<double> normal_dist_3(mean_3[0], cov_3(0, 0));

    std::cout << "mean: \n" << mean_3 << "\ncov: \n" << cov_3 << "\ntime: " << duration << std::endl;

    // Create a Gnuplot object
    Gnuplot gp;

    // Set the title and labels for the plot
    gp << "set title 'Merge Gaussian'\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";

    // Create a vector to store the data points
    std::vector<std::pair<double, double>> data_1, data_2, data_3;

    // Generate function values and store them in the data vector
    for (double x = -10; x <= 10.0; x += 0.01) {
        double y = boost::math::pdf(normal_dist_1, x);
        data_1.push_back(std::make_pair(x, y));

        y = boost::math::pdf(normal_dist_2, x);
        data_2.push_back(std::make_pair(x, y));

        y = boost::math::pdf(normal_dist_3, x);
        data_3.push_back(std::make_pair(x, y));
    }

    // Plot the data
    gp << "plot '-' with lines title 'Function 1', '-' with lines title 'Function 2', '-' with lines title 'Function 3'\n";
    gp.send1d(data_1);
    gp.send1d(data_2);
    gp.send1d(data_3);
}

void merge_inverse_wishart()
{
    double weight_1 = 0.8;
    double v_1 = 6;
    Eigen::Matrix2d V_1 = Eigen::Matrix2d::Identity() * 2;

    double weight_2 = 0.2;
    double v_2 = 5;
    Eigen::Matrix2d V_2 = Eigen::Matrix2d::Identity() * 3;

    double weight[] = {weight_1, weight_2};
    double v[] = {v_1, v_2};
    Eigen::Matrix2d V[] = {V_1, V_2};

    double v_3;
    Eigen::Matrix2d V_3;

    auto start = std::chrono::high_resolution_clock::now();
    merge_inverse_wishart(v_3, V_3, weight, v, V, 2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    std::cout << "v: " << v_3 << "\nV: \n" << V_3 << "\ntime: " << duration << std::endl;

}

int main()
{
    merge_gamma();
    merge_gaussian();
    merge_inverse_wishart();

    return 0;
}