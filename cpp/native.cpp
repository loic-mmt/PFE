#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Vec = std::vector<double>;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
double ratio(double a, double b) { return b != 0 && std::isfinite(b) ? a / b : quiet_nan; }
int sign(double x) { return (x > 0) - (x < 0); }
double mean(const Vec &x) {
    return x.empty() ? quiet_nan : std::accumulate(x.begin(), x.end(), 0.) / x.size();
}
double variance(const Vec &x) {
    if (x.empty())
        return quiet_nan;
    double m = mean(x), sum = 0;
    for (double a : x)
        sum += (a - m) * (a - m);
    return sum / x.size();
}
double entropy(const Vec &x) {
    double total = std::accumulate(x.begin(), x.end(), 0.), value = 0;
    if (!total)
        return quiet_nan;
    for (double a : x)
        if (a > 0) {
            double p = a / total;
            value -= p * std::log(p);
        }
    return value;
}
double regression(const Vec &x, const std::string &key) {
    const auto n = x.size();
    if (n < 2)
        return quiet_nan;
    double mx = (n - 1) / 2., my = mean(x), xy = 0, xx = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = i - mx;
        xy += d * (x[i] - my);
        xx += d * d;
    }
    double slope = xy / xx;
    if (key == "slope")
        return slope;
    double intercept = my - slope * mx;
    if (key == "intercept")
        return intercept;
    double rss = 0, total = 0;
    for (size_t i = 0; i < n; ++i) {
        double r = x[i] - (intercept + slope * i);
        rss += r * r;
        total += (x[i] - my) * (x[i] - my);
    }
    return key == "residual_std" ? std::sqrt(rss / n) : 1 - ratio(rss, total);
}
double correlation(const Vec &x, const Vec &y) {
    Vec a, b;
    for (size_t i = 0; i < x.size(); ++i)
        if (std::isfinite(x[i]) && std::isfinite(y[i])) {
            a.push_back(x[i]);
            b.push_back(y[i]);
        }
    if (a.size() < 2 ||
        *std::min_element(a.begin(), a.end()) == *std::max_element(a.begin(), a.end()) ||
        *std::min_element(b.begin(), b.end()) == *std::max_element(b.begin(), b.end()))
        return quiet_nan;
    double ma = mean(a), mb = mean(b), ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double u = a[i] - ma, v = b[i] - mb;
        ab += u * v;
        aa += u * u;
        bb += v * v;
    }
    return std::clamp(ab / std::sqrt(aa * bb), -1., 1.);
}
double median(Vec &x) {
    auto n = x.size();
    if (!n)
        return quiet_nan;
    auto middle = x.begin() + n / 2;
    std::nth_element(x.begin(), middle, x.end());
    return n % 2 ? *middle : (*middle + *std::max_element(x.begin(), middle)) / 2;
}
void vector_shape(const py::array &x, py::ssize_t n = -1) {
    if (x.ndim() != 1 || (n >= 0 && x.shape(0) != n))
        throw std::invalid_argument("Expected equal-length one-dimensional arrays");
}

struct Profile {
    int bins, segments;
    std::vector<std::uint8_t> presence;
    std::vector<std::int64_t> zones, times;
    Vec centers, counts, tpo, segment_zone, segment_count, segment_center;
    Profile(const double *low, const double *high, const double *close, const std::int64_t *stamp,
            size_t n, std::int64_t start, std::int64_t duration, int b, int s)
        : bins(b), segments(s), presence(size_t(b) * s, 0), zones(n), times(n), centers(b),
          counts(b, 0), tpo(b, 0), segment_zone(s, 0), segment_count(s, 0),
          segment_center(s, quiet_nan) {
        double lo = *std::min_element(low, low + n), hi = *std::max_element(high, high + n);
        double step = (hi - lo) / bins;
        for (int k = 0; k < bins; ++k)
            centers[k] =
                hi == lo ? lo : ((lo + k * step) + (k + 1 == bins ? hi : lo + (k + 1) * step)) / 2;
        auto zone = [&](double value) {
            return hi == lo
                       ? bins / 2
                       : std::clamp(int(std::floor((value - lo) / (hi - lo) * bins)), 0, bins - 1);
        };
        for (size_t i = 0; i < n; ++i) {
            int t = std::clamp(int(std::floor(double(stamp[i] - start) / duration * segments)), 0,
                               segments - 1);
            times[i] = t;
            zones[i] = zone(close[i]);
            counts[zones[i]]++;
            segment_zone[t] += zones[i];
            segment_count[t]++;
            for (int k = zone(low[i]); k <= zone(high[i]); ++k)
                presence[size_t(k) * segments + t] = 1;
        }
        for (int s = 0; s < segments; ++s) {
            double total = 0, count = 0;
            for (int k = 0; k < bins; ++k)
                if (presence[size_t(k) * segments + s]) {
                    tpo[k]++;
                    total += centers[k];
                    count++;
                }
            segment_center[s] = ratio(total, count);
        }
    }
    double get(const std::string &name, double last_close) const {
        auto key = name.substr(8);
        if (key == "close_entropy")
            return entropy(counts);
        if (key == "tpo_entropy")
            return entropy(tpo);
        if (key == "close_concentration") {
            double sum = 0;
            for (double c : counts)
                sum += std::pow(c / zones.size(), 2);
            return sum;
        }
        if (key == "occupied_zones")
            return std::count_if(counts.begin(), counts.end(), [](double c) { return c > 0; });
        if (key == "first_zone")
            return zones.front();
        if (key == "last_zone")
            return zones.back();
        if (key == "tpo_single_block_rows")
            return std::count(tpo.begin(), tpo.end(), 1.);
        if (key == "zone_transitions" || key == "zone_crossings" || key == "zone_revisits" ||
            key == "longest_zone_stay" || key == "mean_zone_stay" || key == "top_to_bottom" ||
            key == "bottom_to_top") {
            double transitions = 0, crossings = 0, runs = 1, longest = 1, current = 1, down = 0,
                   up = 0;
            std::vector<int> visits(bins, 0);
            visits[zones[0]]++;
            for (size_t i = 1; i < zones.size(); ++i) {
                if (zones[i] != zones[i - 1]) {
                    transitions++;
                    runs++;
                    visits[zones[i]]++;
                    current = 1;
                } else
                    current++;
                longest = std::max(longest, current);
                crossings += std::abs(zones[i] - zones[i - 1]);
                down += zones[i - 1] >= bins * 2 / 3. && zones[i] < bins / 3.;
                up += zones[i - 1] < bins / 3. && zones[i] >= bins * 2 / 3.;
            }
            if (key == "zone_transitions")
                return transitions;
            if (key == "zone_crossings")
                return crossings;
            if (key == "longest_zone_stay")
                return longest;
            if (key == "mean_zone_stay")
                return zones.size() / runs;
            if (key == "top_to_bottom")
                return down;
            if (key == "bottom_to_top")
                return up;
            double revisits = 0;
            for (int v : visits)
                revisits += std::max(0, v - 1);
            return revisits;
        }
        if (key.rfind("tpo_poc", 0) == 0 || key.rfind("tpo_value_area", 0) == 0) {
            double largest = *std::max_element(tpo.begin(), tpo.end());
            std::vector<int> tied;
            for (int k = 0; k < bins; ++k)
                if (tpo[k] == largest)
                    tied.push_back(k);
            int poc = tied[(tied.size() - 1) / 2];
            if (key == "tpo_poc")
                return centers[poc];
            if (key == "tpo_poc_close_distance")
                return last_close / centers[poc] - 1;
            int left = poc, right = poc;
            double covered = tpo[poc], total = std::accumulate(tpo.begin(), tpo.end(), 0.);
            while (covered < .7 * total && (left > 0 || right < bins - 1)) {
                double lc = left > 0 ? tpo[left - 1] : -1,
                       rc = right < bins - 1 ? tpo[right + 1] : -1;
                if (lc >= rc && left > 0)
                    covered += tpo[--left];
                if (rc >= lc && right < bins - 1)
                    covered += tpo[++right];
            }
            if (key == "tpo_value_area_low")
                return centers[left];
            if (key == "tpo_value_area_high")
                return centers[right];
            if (key == "tpo_value_area_width")
                return centers[right] - centers[left];
        }
        for (const std::string prefix :
             {"close_share_bin", "tpo_blocks_bin", "tpo_share_bin", "tpo_bin"}) {
            if (key.rfind(prefix, 0) != 0)
                continue;
            int k = std::stoi(key.substr(prefix.size()));
            if (k < 0 || k >= bins)
                break;
            if (prefix == "close_share_bin")
                return counts[k] / zones.size();
            if (prefix == "tpo_blocks_bin")
                return tpo[k];
            if (prefix == "tpo_share_bin")
                return tpo[k] / segments;
            auto pos = key.find("_segment");
            if (pos == std::string::npos)
                break;
            int segment = std::stoi(key.substr(pos + 8));
            if (segment >= 0 && segment < segments)
                return presence[size_t(k) * segments + segment];
        }
        if (key.rfind("segment", 0) == 0) {
            auto pos = key.find('_');
            int s = std::stoi(key.substr(7, pos - 7));
            if (s < 0 || s >= segments)
                throw std::invalid_argument("Invalid profile segment");
            auto field = key.substr(pos + 1);
            if (field == "mean_close_zone")
                return ratio(segment_zone[s], segment_count[s]);
            if (field == "center")
                return segment_center[s];
            if (s && field == "center_change")
                return segment_center[s] - segment_center[s - 1];
            if (s && field == "overlap_previous") {
                double intersection = 0, union_count = 0;
                for (int k = 0; k < bins; ++k) {
                    bool a = presence[size_t(k) * segments + s - 1],
                         b = presence[size_t(k) * segments + s];
                    intersection += a && b;
                    union_count += a || b;
                }
                return ratio(intersection, union_count);
            }
        }
        throw std::invalid_argument("Unknown profile feature: " + name);
    }
};

struct Path {
    const double *o, *h, *l, *c, *v;
    const std::int64_t *t;
    size_t n;
    std::int64_t start, duration, source;
    int bins, segments;
    double threshold;
    bool complete, returns_ready = false;
    Vec r, valid, elapsed, ranges, seconds;
    std::unique_ptr<Profile> profile;
    std::unordered_map<std::string, double> cached;
    Path(const double *op, const double *hi, const double *lo, const double *cl, const double *vol,
         const std::int64_t *time, size_t size, std::int64_t begin, std::int64_t dur,
         std::int64_t step, int b, int s, double cutoff)
        : o(op), h(hi), l(lo), c(cl), v(vol), t(time), n(size), start(begin), duration(dur),
          source(step), bins(b), segments(s), threshold(cutoff) {
        complete = n == size_t(duration / source) && t[0] == start;
        for (size_t i = 1; i < n; ++i)
            if (t[i] - t[i - 1] != source)
                complete = false;
    }
    void returns() {
        if (returns_ready)
            return;
        r.resize(n);
        r[0] = std::log(c[0] / o[0]);
        double previous = std::log(c[0]);
        for (size_t i = 1; i < n; ++i) {
            double current = std::log(c[i]);
            r[i] = t[i] - t[i - 1] == source ? current - previous : quiet_nan;
            previous = current;
        }
        for (double a : r)
            if (std::isfinite(a))
                valid.push_back(a);
        returns_ready = true;
    }
    const Vec &time_fraction() {
        if (elapsed.empty()) {
            elapsed.resize(n);
            for (size_t i = 0; i < n; ++i)
                elapsed[i] = double(t[i] - start) / duration;
        }
        return elapsed;
    }
    const Vec &range_values() {
        if (ranges.empty()) {
            ranges.resize(n);
            for (size_t i = 0; i < n; ++i)
                ranges[i] = h[i] - l[i];
        }
        return ranges;
    }
    double excursion(const std::string &key) {
        bool invert = key.find("drawup") != std::string::npos;
        std::string prefix = invert ? "drawup" : "drawdown";
        if (cached.count("max_" + prefix))
            return cached.at(key);
        Vec path(n + 1), sec(n + 1), depth(n + 1), peaks(n + 1);
        path[0] = invert ? 1 / o[0] : o[0];
        sec[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            path[i + 1] = invert ? 1 / c[i] : c[i];
            sec[i + 1] = double(t[i] - start + source) / 1e9;
        }
        double peak_value = path[0];
        size_t trough = 0;
        double episodes = 0, under_count = 0, under_seconds = 0, longest_seconds = 0,
               active_seconds = 0, longest_run = 0, active_run = 0;
        for (size_t i = 0; i <= n; ++i) {
            peak_value = std::max(peak_value, path[i]);
            peaks[i] = peak_value;
            depth[i] = 1 - path[i] / peak_value;
            if (depth[i] > depth[trough])
                trough = i;
            episodes += depth[i] >= threshold && (i == 0 || depth[i - 1] < threshold);
            bool under = depth[i] > 0;
            under_count += under;
            active_run = under ? active_run + 1 : 0;
            longest_run = std::max(longest_run, active_run);
            if (i) {
                double dt = sec[i] - sec[i - 1];
                under_seconds += under ? dt : 0;
                active_seconds = under ? active_seconds + dt : 0;
                longest_seconds = std::max(longest_seconds, active_seconds);
            }
        }
        size_t peak = 0;
        for (size_t i = 0; i <= trough; ++i)
            if (path[i] == peaks[trough])
                peak = i;
        size_t recovery = n + 1;
        for (size_t i = trough + 1; i <= n; ++i)
            if (path[i] >= peaks[trough]) {
                recovery = i;
                break;
            }
        cached["max_" + prefix] = depth[trough];
        cached[prefix + "_peak_to_trough_seconds"] = sec[trough] - sec[peak];
        cached[prefix + "_recovery_seconds"] =
            recovery <= n ? sec[recovery] - sec[trough] : quiet_nan;
        cached[prefix + "_recovered"] = recovery <= n || depth[trough] == 0;
        cached[prefix + "_recovery_fraction"] =
            ratio(path.back() - path[trough], peaks[trough] - path[trough]);
        cached[prefix + "_threshold_episodes"] = episodes;
        cached[prefix + "_underwater_observations"] = under_count;
        cached[prefix + "_underwater_seconds"] = under_seconds;
        cached[prefix + "_longest_underwater_seconds"] = longest_seconds;
        cached[prefix + "_longest_underwater_run"] = longest_run;
        return cached.at(key);
    }
    double segment(const std::string &key) {
        if (!cached.count("segment0_volume")) {
            returns();
            time_fraction();
            double total = std::accumulate(v, v + n, 0.), last_return = quiet_nan,
                   last_volatility = quiet_nan, last_volume = quiet_nan;
            for (int s = 0; s < segments; ++s) {
                size_t first = n, last = 0;
                double rv = 0, volume = 0, hi = -INFINITY, lo = INFINITY;
                for (size_t i = 0; i < n; ++i)
                    if (std::clamp(int(std::floor(elapsed[i] * segments)), 0, segments - 1) == s) {
                        first = std::min(first, i);
                        last = i;
                        volume += v[i];
                        hi = std::max(hi, h[i]);
                        lo = std::min(lo, l[i]);
                        if (std::isfinite(r[i]))
                            rv += r[i] * r[i];
                    }
                auto prefix = "segment" + std::to_string(s) + "_";
                if (first == n) {
                    for (auto field : {"return", "range", "realized_variance", "volatility",
                                       "volume", "volume_share", "close_position", "return_change",
                                       "volatility_change", "volume_change"})
                        cached[prefix + field] = quiet_nan;
                    continue;
                }
                double ret = std::log(c[last] / o[first]), volatility = std::sqrt(rv);
                cached[prefix + "return"] = ret;
                cached[prefix + "range"] = hi - lo;
                cached[prefix + "realized_variance"] = rv;
                cached[prefix + "volatility"] = volatility;
                cached[prefix + "volume"] = volume;
                cached[prefix + "volume_share"] = ratio(volume, total);
                cached[prefix + "close_position"] = ratio(c[last] - lo, hi - lo);
                cached[prefix + "return_change"] = ret - last_return;
                cached[prefix + "volatility_change"] = volatility - last_volatility;
                cached[prefix + "volume_change"] = volume - last_volume;
                last_return = ret;
                last_volatility = volatility;
                last_volume = volume;
            }
        }
        auto it = cached.find(key);
        if (it == cached.end())
            throw std::invalid_argument("Unknown segment feature: " + key);
        return it->second;
    }
    double get(const std::string &name) {
        if (name.rfind("profile.", 0) == 0) {
            if (!profile)
                profile = std::make_unique<Profile>(l, h, c, t, n, start, duration, bins, segments);
            return profile->get(name, c[n - 1]);
        }
        auto key = name.substr(name.find('.') + 1);
        if (cached.count(key))
            return cached.at(key);
        if (key.find("drawdown") != std::string::npos || key.find("drawup") != std::string::npos)
            return excursion(key);
        if (key.rfind("segment", 0) == 0)
            return segment(key);
        if (key == "end_start_volatility_ratio")
            return ratio(segment("segment" + std::to_string(segments - 1) + "_volatility"),
                         segment("segment0_volatility"));
        if (key == "range_mean")
            return mean(range_values());
        if (key == "range_std")
            return std::sqrt(variance(range_values()));
        if (key == "range_expansion_slope")
            return regression(range_values(), "slope");
        if (key.rfind("regression_", 0) == 0 || key == "curvature" || key == "signed_area" ||
            key == "absolute_area" || key == "straight_line_error") {
            Vec x(n);
            for (size_t i = 0; i < n; ++i)
                x[i] = std::log(c[i] / o[0]);
            if (key.rfind("regression_", 0) == 0)
                return regression(x, key.substr(11));
            if (key == "curvature") {
                if (n < 3)
                    return quiet_nan;
                double sum = 0;
                for (size_t i = 2; i < n; ++i)
                    sum += (x[i] - x[i - 1]) - (x[i - 1] - x[i - 2]);
                return sum / (n - 2);
            }
            double sum = 0;
            for (size_t i = 0; i < n; ++i) {
                double value = c[i] / o[0] - 1;
                if (key == "absolute_area")
                    value = std::abs(value);
                if (key == "straight_line_error") {
                    double line = std::log(c[n - 1] / o[0]) *
                                  std::clamp(double(t[i] - start + source) / duration, 0., 1.);
                    value = (x[i] - line) * (x[i] - line);
                }
                sum += value;
            }
            return key == "straight_line_error" ? std::sqrt(sum / n) : sum / n;
        }
        if ((key.rfind("high_", 0) == 0 && key.rfind("high_volume_", 0) != 0) ||
            key.rfind("low_", 0) == 0 || key == "extrema_delay_fraction") {
            bool hi = key.rfind("low_", 0) != 0;
            const double *values = hi ? h : l;
            double extreme =
                hi ? *std::max_element(values, values + n) : *std::min_element(values, values + n);
            size_t first = n, last = 0, count = 0;
            for (size_t i = 0; i < n; ++i)
                if (values[i] == extreme) {
                    first = std::min(first, i);
                    last = i;
                    count++;
                }
            if (key == "high_before_low" || key == "extrema_delay_fraction") {
                size_t high = std::max_element(h, h + n) - h, low = std::min_element(l, l + n) - l;
                return key == "high_before_low"
                           ? (high == low ? quiet_nan : double(high < low))
                           : double(t[std::max(high, low)] - t[std::min(high, low)]) / duration;
            }
            if (key.find("first_time") != std::string::npos)
                return double(t[first] - start) / duration;
            if (key.find("last_time") != std::string::npos)
                return double(t[last] - start) / duration;
            if (key.find("to_close") != std::string::npos)
                return double(t[n - 1] - t[last]) / 1e9;
            if (key.find("retests") != std::string::npos)
                return count - 1;
            // high_volume fields belong to the volume section below.
        }
        if (key == "observed_bars")
            return n;
        if (key == "expected_bars")
            return duration / source;
        if (key == "missing_bars")
            return std::max(0., double(duration / source) - n);
        if (key == "coverage")
            return double(n) / (duration / source);
        if (key == "complete")
            return complete;
        if (key == "observed_span_seconds")
            return double(t[n - 1] - t[0] + source) / 1e9;
        if (key == "largest_gap_seconds") {
            double gap = n == 1 ? source : 0;
            for (size_t i = 1; i < n; ++i)
                gap = std::max(gap, double(t[i] - t[i - 1]));
            return gap / 1e9;
        }
        if (key == "unchanged_close_fraction") {
            if (n == 1)
                return quiet_nan;
            double same = 0;
            for (size_t i = 1; i < n; ++i)
                same += c[i] == c[i - 1];
            return same / (n - 1);
        }
        if (key == "open_crossings") {
            double count = 0;
            for (size_t i = 1; i < n; ++i)
                count += (c[i] > o[0]) != (c[i - 1] > o[0]);
            return count;
        }
        if (key == "price_quadratic_variation") {
            if (!complete)
                return quiet_nan;
            double sum = 0, previous = o[0];
            for (size_t i = 0; i < n; ++i) {
                sum += (c[i] - previous) * (c[i] - previous);
                previous = c[i];
            }
            return sum;
        }
        returns();
        if (key == "return_coverage")
            return double(valid.size()) / n;
        double rv = 0, length = 0, sum = 0, up = 0, down = 0, up_sum = 0, down_sum = 0;
        for (double a : valid) {
            rv += a * a;
            length += std::abs(a);
            sum += a;
            if (a > 0) {
                up += a * a;
                up_sum += a;
            }
            if (a < 0) {
                down += a * a;
                down_sum += a;
            }
        }
        if (key == "realized_variance" || key == "realized_volatility")
            return complete ? (key == "realized_variance" ? rv : std::sqrt(rv)) : quiet_nan;
        if (key == "realized_variance_observed")
            return rv;
        if (key == "absolute_return_sum")
            return length;
        if (key == "up_semivariance")
            return up;
        if (key == "down_semivariance")
            return down;
        if (key == "semivariance_ratio")
            return ratio(up, down);
        if (key == "up_return_sum")
            return up_sum;
        if (key == "down_return_sum")
            return down_sum;
        if (key == "directional_efficiency")
            return ratio(std::abs(sum), length);
        if (key == "directional_imbalance" || key == "up_fraction" || key == "down_fraction" ||
            key == "flat_fraction") {
            double positives = 0, negatives = 0;
            for (double a : valid) {
                positives += a > 0;
                negatives += a < 0;
            }
            return key == "directional_imbalance" ? (positives - negatives) / valid.size()
                   : key == "up_fraction"         ? positives / valid.size()
                   : key == "down_fraction"       ? negatives / valid.size()
                                                  : 1 - (positives + negatives) / valid.size();
        }
        if (key == "max_positive_return")
            return std::max(0., *std::max_element(valid.begin(), valid.end()));
        if (key == "max_negative_return")
            return std::min(0., *std::min_element(valid.begin(), valid.end()));
        if (key == "max_absolute_return" || key == "large_move_count") {
            double value = 0;
            for (double a : valid)
                value = key == "large_move_count" ? value + (std::abs(a) >= threshold)
                                                  : std::max(value, std::abs(a));
            return value;
        }
        if (key == "return_energy_concentration") {
            double e4 = 0;
            for (double a : valid)
                e4 += a * a * a * a;
            return ratio(e4, rv * rv);
        }
        if (key.rfind("top", 0) == 0 && key.find("_energy_share") != std::string::npos) {
            int k = std::stoi(key.substr(3));
            Vec energy;
            for (double a : valid)
                energy.push_back(a * a);
            std::sort(energy.begin(), energy.end(), std::greater<double>());
            return ratio(std::accumulate(energy.begin(),
                                         energy.begin() + std::min(size_t(k), energy.size()), 0.),
                         rv);
        }
        if (key == "turn_count" || key == "mean_run_length" || key == "longest_up_run" ||
            key == "longest_down_run") {
            int previous = sign(valid[0]), last_nonzero = 0;
            double runs = 1, current = 0, longest_up = 0, longest_down = 0, turns = 0;
            for (size_t i = 0; i < valid.size(); ++i) {
                int s = sign(valid[i]);
                if (i && s != previous) {
                    runs++;
                    current = 0;
                }
                current++;
                if (s == 1)
                    longest_up = std::max(longest_up, current);
                if (s == -1)
                    longest_down = std::max(longest_down, current);
                if (s) {
                    turns += last_nonzero != 0 && last_nonzero != s;
                    last_nonzero = s;
                }
                previous = s;
            }
            return key == "turn_count"        ? turns
                   : key == "mean_run_length" ? valid.size() / runs
                   : key == "longest_up_run"  ? longest_up
                                              : longest_down;
        }
        if (key == "absolute_return_slope") {
            Vec x = valid;
            for (double &a : x)
                a = std::abs(a);
            return regression(x, "slope");
        }
        if (key == "largest_move_time_fraction" || key == "price_volume_peak_delay_fraction") {
            size_t largest = 0;
            for (size_t i = 1; i < n; ++i)
                if (std::isfinite(r[i]) && std::abs(r[i]) > std::abs(r[largest]))
                    largest = i;
            if (key == "largest_move_time_fraction")
                return double(t[largest] - start) / duration;
            size_t volume = std::max_element(v, v + n) - v;
            return double(t[volume] - t[largest]) / duration;
        }
        if (key == "first_half_energy_share" || key == "last_half_energy_share" ||
            key == "half_direction_agreement") {
            double a = 0, b = 0, sa = 0, sb = 0;
            for (size_t i = 0; i < n; ++i)
                if (std::isfinite(r[i])) {
                    if (double(t[i] - start) / duration < .5) {
                        a += r[i] * r[i];
                        sa += r[i];
                    } else {
                        b += r[i] * r[i];
                        sb += r[i];
                    }
                }
            return key == "half_direction_agreement"
                       ? double(sign(sa) == sign(sb))
                       : ratio(key == "first_half_energy_share" ? a : b, rv);
        }
        if (key == "return_autocorr" || key == "absolute_return_autocorr") {
            Vec a(r.begin(), r.end() - 1), b(r.begin() + 1, r.end());
            if (key == "absolute_return_autocorr") {
                for (double &x : a)
                    x = std::abs(x);
                for (double &x : b)
                    x = std::abs(x);
            }
            return correlation(a, b);
        }
        double total = std::accumulate(v, v + n, 0.);
        if (key == "volume_concentration") {
            double sum = 0;
            for (size_t i = 0; i < n; ++i)
                sum += v[i] * v[i];
            return ratio(sum, total * total);
        }
        if (key == "volume_entropy")
            return entropy(Vec(v, v + n));
        if (key == "volume_peak_time_fraction")
            return double(t[std::max_element(v, v + n) - v] - start) / duration;
        if (key == "volume_time_center") {
            double sum = 0;
            for (size_t i = 0; i < n; ++i)
                sum += v[i] * double(t[i] - start) / duration;
            return ratio(sum, total);
        }
        if (key == "volume_slope")
            return regression(Vec(v, v + n), "slope");
        if (key == "zero_volume_count")
            return std::count(v, v + n, 0.);
        if (key == "return_volume_corr" || key == "absolute_return_volume_corr" ||
            key == "range_volume_corr" || key.rfind("return_volume_corr_lag", 0) == 0) {
            int lag = key.rfind("return_volume_corr_lag", 0) == 0 ? std::stoi(key.substr(22)) : 0;
            if (size_t(lag) >= n)
                return quiet_nan;
            Vec a = key == "range_volume_corr" ? range_values() : r;
            if (key == "absolute_return_volume_corr")
                for (double &x : a)
                    x = std::abs(x);
            return correlation(Vec(a.begin() + lag, a.end()), Vec(v, v + n - lag));
        }
        if (key == "volume_up_share" || key == "volume_down_share" ||
            key == "volume_weighted_return") {
            double value = 0, valid_volume = 0;
            for (size_t i = 0; i < n; ++i) {
                if (key == "volume_up_share" && r[i] > 0)
                    value += v[i];
                if (key == "volume_down_share" && r[i] < 0)
                    value += v[i];
                if (std::isfinite(r[i])) {
                    valid_volume += v[i];
                    if (key == "volume_weighted_return")
                        value += v[i] * r[i];
                }
            }
            return ratio(value, key == "volume_weighted_return" ? valid_volume : total);
        }
        if (key.rfind("high_volume_", 0) == 0) {
            double threshold = 2 * total / n, count = 0, episodes = 0, current = 0, longest = 0;
            bool previous = false;
            for (size_t i = 0; i < n; ++i) {
                bool high = v[i] > threshold;
                count += high;
                episodes += high && !previous;
                current = high ? current + 1 : 0;
                longest = std::max(longest, current);
                previous = high;
            }
            return key == "high_volume_episode_count" ? episodes
                   : key == "high_volume_longest_run" ? longest
                                                      : count;
        }
        if (key == "large_move_low_volume_count") {
            Vec volumes(v, v + n);
            double med = median(volumes), count = 0;
            for (size_t i = 0; i < n; ++i)
                count += std::abs(r[i]) >= threshold && v[i] < med;
            return count;
        }
        if (key.rfind("scale", 0) == 0) {
            size_t split = key.find('_');
            int scale = std::stoi(key.substr(5, split - 5));
            if (scale < 1)
                throw std::invalid_argument("scale must be positive");
            auto field = key.substr(split + 1);
            auto expected = duration / source;
            if (!complete || expected % scale || scale > expected)
                return quiet_nan;
            Vec coarse;
            double previous = std::log(o[0]);
            for (size_t i = scale - 1; i < n; i += scale) {
                double current = std::log(c[i]);
                coarse.push_back(current - previous);
                previous = current;
            }
            double value = 0, turns = 0, agreement = 0;
            int last = 0, direction = sign(std::log(c[n - 1] / o[0]));
            for (double a : coarse) {
                value += a * a;
                int s = sign(a);
                agreement += s == direction;
                if (s) {
                    turns += last != 0 && last != s;
                    last = s;
                }
            }
            if (field == "realized_variance")
                return value;
            if (field == "rv_ratio")
                return ratio(value, rv);
            if (field == "turn_count")
                return turns;
            if (field == "direction_agreement")
                return agreement / coarse.size();
        }
        throw std::invalid_argument("Unknown intrabar feature: " + name);
    }
};

const std::vector<std::string> window_keys = {"skew",
                                              "kurtosis",
                                              "mad",
                                              "regression_slope",
                                              "regression_intercept",
                                              "regression_r_squared",
                                              "regression_residual_std",
                                              "high_age",
                                              "low_age",
                                              "max_drawdown",
                                              "turn_count",
                                              "longest_up_run",
                                              "longest_down_run",
                                              "dot",
                                              "sample_entropy",
                                              "permutation_entropy",
                                              "sign_entropy",
                                              "zero_crossings",
                                              "petrosian_dimension",
                                              "tail_fraction",
                                              "expected_shortfall_05",
                                              "hurst",
                                              "trend_slope",
                                              "wavelet_detail_energy_l1",
                                              "wavelet_detail_energy_l2",
                                              "wavelet_detail_energy_l3",
                                              "wavelet_detail_first_l1",
                                              "wavelet_detail_first_l2",
                                              "wavelet_detail_first_l3"};

// NumPy's contiguous float64 reduction order: preserve constant-window
// rounding/NaN conventions rather than changing standardized moments.
double pairwise_sum(const double *p, size_t n) {
    if (n < 8) {
        double sum = -0.0;
        for (size_t i = 0; i < n; ++i)
            sum += p[i];
        return sum;
    }
    if (n <= 128) {
        double sums[8];
        for (size_t k = 0; k < 8; ++k)
            sums[k] = p[k];
        size_t i = 8;
        for (; i + 7 < n; i += 8)
            for (size_t k = 0; k < 8; ++k)
                sums[k] += p[i + k];
        double sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                     ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < n; ++i)
            sum += p[i];
        return sum;
    }
    size_t split = n / 2;
    split -= split % 8;
    return pairwise_sum(p, split) + pairwise_sum(p + split, n - split);
}

double window_value(const double *p, size_t n, const std::string &key, const Vec &weights,
                    Vec &scratch) {
    if (key == "dot") {
        double sum = 0;
        for (size_t i = 0; i < n; ++i)
            sum += p[i] * weights[i];
        return sum;
    }
    if (key == "high_age" || key == "low_age") {
        size_t last = 0;
        for (size_t i = 1; i < n; ++i)
            if (key == "high_age" ? p[i] >= p[last] : p[i] <= p[last])
                last = i;
        return n - 1 - last;
    }
    if (key == "max_drawdown") {
        double peak = p[0], dd = 0;
        for (size_t i = 0; i < n; ++i) {
            peak = std::max(peak, p[i]);
            dd = std::min(dd, p[i] / peak - 1);
        }
        return dd;
    }
    if (key == "turn_count" || key == "longest_up_run" || key == "longest_down_run") {
        double changes = 0, current = 0, longest = 0;
        int previous = 0;
        for (size_t i = 0; i < n; ++i) {
            int value = sign(p[i]);
            if (key == "turn_count") {
                if (value) {
                    changes += previous && previous != value;
                    previous = value;
                }
            } else {
                int direction = key == "longest_up_run" ? 1 : -1;
                current = value == direction ? current + 1 : 0;
                longest = std::max(longest, current);
            }
        }
        return key == "turn_count" ? changes : longest;
    }
    if (key == "mad") {
        scratch.assign(p, p + n);
        double med = median(scratch);
        for (size_t i = 0; i < n; ++i)
            scratch[i] = std::abs(p[i] - med);
        return median(scratch);
    }
    if (key.rfind("regression_", 0) == 0 || key == "trend_slope")
        return regression(Vec(p, p + n), key == "trend_slope" ? "slope" : key.substr(11));
    double avg = pairwise_sum(p, n) / n;
    scratch.resize(n);
    for (size_t i = 0; i < n; ++i)
        scratch[i] = (p[i] - avg) * (p[i] - avg);
    double m2 = pairwise_sum(scratch.data(), n) / n;
    if (key == "skew" || key == "kurtosis") {
        if (!m2 || n < size_t(key == "skew" ? 3 : 4))
            return quiet_nan;
        for (size_t i = 0; i < n; ++i) {
            double d = p[i] - avg;
            scratch[i] = key == "skew" ? d * d * d : d * d * d * d;
        }
        double moment = pairwise_sum(scratch.data(), n) / n;
        double size = double(n);
        return key == "skew"
                   ? std::sqrt(size * (size - 1)) / (size - 2) * moment / std::pow(m2, 1.5)
                   : (size - 1) / ((size - 2) * (size - 3)) *
                         ((size + 1) * moment / (m2 * m2) - 3 * (size - 1));
    }
    if (key == "sample_entropy") {
        if (n < 4)
            return quiet_nan;
        if (!m2)
            return 0;
        double tolerance = .2 * std::sqrt(m2), a = 0, b = 0;
        for (size_t i = 0; i < n - 2; ++i)
            for (size_t j = i + 1; j < n - 2; ++j)
                if (std::abs(p[i] - p[j]) <= tolerance &&
                    std::abs(p[i + 1] - p[j + 1]) <= tolerance) {
                    a++;
                    b += std::abs(p[i + 2] - p[j + 2]) <= tolerance;
                }
        return a && b ? -std::log(b / a) : quiet_nan;
    }
    if (key == "permutation_entropy") {
        if (n < 3)
            return quiet_nan;
        Vec counts(6, 0);
        for (size_t i = 0; i < n - 2; ++i) {
            std::array<int, 3> ids{0, 1, 2};
            std::stable_sort(ids.begin(), ids.end(),
                             [&](int a, int b) { return p[i + a] < p[i + b]; });
            int rank = 0;
            for (int k = 1; k < 3; ++k)
                if (ids[k] < ids[0])
                    rank += 2;
            if (ids[2] < ids[1])
                rank++;
            counts[rank]++;
        }
        return entropy(counts);
    }
    if (key == "sign_entropy") {
        Vec counts(3, 0);
        for (size_t i = 0; i < n; ++i)
            counts[sign(p[i]) + 1]++;
        return entropy(counts);
    }
    if (key == "zero_crossings") {
        double count = 0;
        for (size_t i = 1; i < n; ++i)
            count += sign(p[i] - avg) != sign(p[i - 1] - avg);
        return count;
    }
    if (key == "petrosian_dimension") {
        if (n < 3)
            return quiet_nan;
        double changes = 0;
        for (size_t i = 2; i < n; ++i)
            changes += sign(p[i] - p[i - 1]) != sign(p[i - 1] - p[i - 2]);
        double size = n;
        return std::log10(size) / (std::log10(size) + std::log10(size / (size + .4 * changes)));
    }
    if (key == "tail_fraction") {
        if (!m2)
            return 0;
        double count = 0;
        for (size_t i = 0; i < n; ++i)
            count += std::abs(p[i] - avg) > 2 * std::sqrt(m2);
        return count / n;
    }
    if (key == "expected_shortfall_05") {
        scratch.assign(p, p + n);
        std::sort(scratch.begin(), scratch.end());
        double pos = .05 * (n - 1);
        size_t i = size_t(pos);
        double q = scratch[i] + (pos - i) * (scratch[std::min(i + 1, n - 1)] - scratch[i]), sum = 0,
               count = 0;
        for (size_t j = 0; j < n; ++j)
            if (p[j] <= q) {
                sum += p[j];
                count++;
            }
        return ratio(sum, count);
    }
    if (key == "hurst") {
        if (n < 8)
            return quiet_nan;
        Vec x, y;
        for (size_t lag = 2; lag <= std::min(size_t(20), n / 2); ++lag) {
            scratch.resize(n - lag);
            for (size_t i = lag; i < n; ++i)
                scratch[i - lag] = p[i] - p[i - lag];
            double scale = std::sqrt(variance(scratch));
            if (scale > 0) {
                x.push_back(std::log(double(lag)));
                y.push_back(std::log(scale));
            }
        }
        if (x.size() < 2)
            return quiet_nan;
        double mx = mean(x), my = mean(y), xx = 0, xy = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            xx += (x[i] - mx) * (x[i] - mx);
            xy += (x[i] - mx) * (y[i] - my);
        }
        return xy / xx;
    }
    if (key.rfind("wavelet_detail_", 0) == 0) {
        int level = key.back() - '0';
        Vec approximation(p, p + n), detail;
        for (int k = 1; k <= level; ++k) {
            if (approximation.size() < 2)
                return quiet_nan;
            size_t length = (approximation.size() + 1) / 2;
            Vec next(length);
            detail.resize(length);
            for (size_t i = 0; i < length; ++i) {
                double a = approximation[2 * i],
                       b = approximation[std::min(2 * i + 1, approximation.size() - 1)];
                next[i] = (a + b) * std::sqrt(.5);
                detail[i] = (a - b) * std::sqrt(.5);
            }
            approximation = std::move(next);
        }
        if (key.rfind("wavelet_detail_first", 0) == 0)
            return detail[0];
        double sum = 0;
        for (double d : detail)
            sum += d * d;
        return sum / detail.size();
    }
    throw std::invalid_argument("Unknown rolling kernel: " + key);
}

py::array_t<double> rolling(Doubles input, int window, const std::vector<std::string> &keys,
                            Doubles weight_array) {
    vector_shape(input);
    vector_shape(weight_array);
    if (window < 1)
        throw std::invalid_argument("window must be positive");
    for (const auto &key : keys)
        if (std::find(window_keys.begin(), window_keys.end(), key) == window_keys.end())
            throw std::invalid_argument("Unknown rolling kernel: " + key);
    if (std::find(keys.begin(), keys.end(), "dot") != keys.end() && weight_array.size() != window)
        throw std::invalid_argument("weights must match window");
    const auto n = input.size();
    py::array_t<double> result({py::ssize_t(keys.size()), n});
    double *out = result.mutable_data();
    const double *values = input.data();
    Vec weights(weight_array.data(), weight_array.data() + weight_array.size()), scratch;
    scratch.reserve(window);
    {
        py::gil_scoped_release release;
        std::fill(out, out + keys.size() * n, quiet_nan);
        int invalid = 0;
        for (py::ssize_t end = 0; end < n; ++end) {
            invalid += !std::isfinite(values[end]);
            if (end >= window)
                invalid -= !std::isfinite(values[end - window]);
            if (end + 1 < window || invalid)
                continue;
            for (size_t k = 0; k < keys.size(); ++k)
                out[k * n + end] =
                    window_value(values + end - window + 1, window, keys[k], weights, scratch);
        }
    }
    return result;
}

py::tuple profile_arrays(Doubles low, Doubles high, Doubles close, Ints timestamps,
                         std::int64_t start, std::int64_t duration, int bins, int segments) {
    vector_shape(low);
    auto n = low.size();
    vector_shape(high, n);
    vector_shape(close, n);
    vector_shape(timestamps, n);
    if (!n || bins < 1 || segments < 1 || duration < 1)
        throw std::invalid_argument(
            "Non-empty profile, positive bins, segments and duration required");
    py::array_t<bool> presence({py::ssize_t(bins), py::ssize_t(segments)});
    Ints zones(n), times(n);
    Doubles centers(bins);
    bool *pp = presence.mutable_data();
    auto *zz = zones.mutable_data(), *tt = times.mutable_data();
    auto *cc = centers.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i)
            if (!std::isfinite(low.data()[i]) || !std::isfinite(high.data()[i]) ||
                !std::isfinite(close.data()[i]) || low.data()[i] > high.data()[i] ||
                timestamps.data()[i] < start || timestamps.data()[i] - start >= duration)
                throw std::invalid_argument("Invalid profile values or time window");
        Profile profile(low.data(), high.data(), close.data(), timestamps.data(), n, start,
                        duration, bins, segments);
        for (size_t i = 0; i < profile.presence.size(); ++i)
            pp[i] = profile.presence[i];
        std::copy(profile.zones.begin(), profile.zones.end(), zz);
        std::copy(profile.times.begin(), profile.times.end(), tt);
        std::copy(profile.centers.begin(), profile.centers.end(), cc);
    }
    return py::make_tuple(presence, zones, centers, times);
}

py::array_t<double> intrabar_batch(Doubles o, Doubles h, Doubles l, Doubles c, Doubles v,
                                   Ints timestamps, Ints first, Ints ends, Ints starts,
                                   std::int64_t duration, std::int64_t source, int bins,
                                   int segments, double threshold,
                                   const std::vector<std::string> &names) {
    vector_shape(o);
    auto n = o.size();
    for (const auto &a : {h, l, c, v})
        vector_shape(a, n);
    vector_shape(timestamps, n);
    vector_shape(first);
    auto groups = first.size();
    vector_shape(ends, groups);
    vector_shape(starts, groups);
    if (duration <= 0 || source <= 0 || duration % source || bins < 1 || segments < 1 ||
        !std::isfinite(threshold) || threshold <= 0)
        throw std::invalid_argument("Invalid intrabar configuration");
    py::array_t<double> result({py::ssize_t(names.size()), groups});
    double *out = result.mutable_data();
    const auto *ff = first.data(), *ee = ends.data(), *ss = starts.data(), *tt = timestamps.data();
    const auto *op = o.data(), *hi = h.data(), *lo = l.data(), *cl = c.data(), *vol = v.data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i)
            if (!std::isfinite(op[i]) || !std::isfinite(hi[i]) || !std::isfinite(lo[i]) ||
                !std::isfinite(cl[i]) || !std::isfinite(vol[i]) || op[i] <= 0 || lo[i] <= 0 ||
                cl[i] <= 0 || hi[i] < std::max(op[i], cl[i]) || lo[i] > std::min(op[i], cl[i]) ||
                vol[i] < 0)
                throw std::invalid_argument("Invalid OHLCV values");
        std::int64_t previous = 0;
        for (py::ssize_t g = 0; g < groups; ++g) {
            auto left = ff[g], right = ee[g];
            if (left < previous || left < 0 || right <= left || right > n)
                throw std::invalid_argument("Invalid intrabar boundaries");
            previous = right;
            for (auto i = left; i < right; ++i)
                if (tt[i] < ss[g] || tt[i] - ss[g] >= duration || (i > left && tt[i] <= tt[i - 1]))
                    throw std::invalid_argument("Invalid intrabar timestamps");
            Path path(op + left, hi + left, lo + left, cl + left, vol + left, tt + left,
                      right - left, ss[g], duration, source, bins, segments, threshold);
            for (size_t k = 0; k < names.size(); ++k)
                out[k * groups + g] = path.get(names[k]);
        }
    }
    return result;
}

PYBIND11_MODULE(_native, m) {
    m.doc() = "PFE C++17 kernels: batched intrabar, TPO reconstruction and rolling descriptors.";
    m.def("profile_arrays", &profile_arrays);
    m.def("intrabar_batch", &intrabar_batch);
    m.def("rolling", &rolling);
    m.attr("window_keys") = window_keys;
}
