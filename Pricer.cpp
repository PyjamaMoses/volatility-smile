#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

class BouchardSornett {
public:
    BouchardSornett(const std::vector<double>& prices)
        : prices(prices), n(prices.size()), epsilon(1e-10) {}

    std::vector<double> calculate_returnrate(int interval = 1) {
        returnrate.resize(n - interval);
        for (size_t i = 0; i < n - interval; ++i) {
            returnrate[i] = (prices[i + interval] - prices[i]) / (prices[i] + epsilon);
        }
        return returnrate;
    }

    std::vector<double> detrended_returnrate() {
        calculate_returnrate();
        double mean = std::accumulate(returnrate.begin(), returnrate.end(), 0.0) / returnrate.size();
        detrended.resize(returnrate.size());
        for (size_t i = 0; i < returnrate.size(); ++i) {
            detrended[i] = returnrate[i] - mean;
        }
        return detrended;
    }

    std::map<double, double> calculate_pdf(bool ifplot = false, int nbin = 0) {
        detrended_returnrate();
        double Rmax = *std::max_element(detrended.begin(), detrended.end());
        double Rmin = *std::min_element(detrended.begin(), detrended.end());
        if (nbin == 0) {
            nbin = static_cast<int>(std.sqrt(n));
        }
        double step = (Rmax - Rmin) / nbin;
        std::vector<int> counts(nbin, 0);
        for (double R : detrended) {
            int bin = static_cast<int>((R - Rmin) / step);
            if (bin < nbin) {
                counts[bin]++;
            }
        }
        std::map<double, double> pdf_series;
        for (int i = 0; i < nbin; ++i) {
            double R_i = Rmin + i * step;
            pdf_series[R_i] = static_cast<double>(counts[i]) / n;
        }

        if (ifplot) {
            std::vector<double> x, y;
            for (const auto& pair : pdf_series) {
                x.push_back(pair.first);
                y.push_back(pair.second);
            }
            plt::figure_size(10, 6);
            plt::bar(x, y, step, "b");
            plt::title("Probability Density Function (PDF)");
            plt::xlabel("Detrended Return Rate");
            plt::ylabel("Probability");
            plt::grid(true);
            plt::show();
        }

        return pdf_series;
    }

private:
    std::vector<double> prices;
    size_t n;
    double epsilon;
    std::vector<double> returnrate;
    std::vector<double> detrended;
};

int main() {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    BouchardSornett bs(prices);
    auto pdf = bs.calculate_pdf(true);
    return 0;
}