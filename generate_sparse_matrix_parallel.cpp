#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <random>
#include <omp.h>
#include <set>
#include <chrono>

void generateRandomSparseMatrix(const std::string& filename, int num_rows, int num_cols, int num_nonzeros) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write Matrix Market header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << num_rows << " " << num_cols << " " << num_nonzeros << "\n";

    std::vector<std::tuple<int, int, double>> elements;
    elements.reserve(num_nonzeros);

    int num_threads = omp_get_max_threads();
    int nonzeros_per_thread = num_nonzeros / num_threads;

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ omp_get_thread_num());
        std::uniform_int_distribution<> dis_col(0, num_cols - 1);
        std::uniform_real_distribution<> dis_val(0.0, 100.0);

        int thread_id = omp_get_thread_num();
        int rows_per_thread = num_rows / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = (thread_id == num_threads - 1) ? num_rows : start_row + rows_per_thread;

        std::set<std::pair<int, int>> local_used_positions;
        std::vector<std::tuple<int, int, double>> local_elements;

        for (int i = 0; i < nonzeros_per_thread; ++i) {
            int row, col;
            do {
                row = start_row + (gen() % (end_row - start_row));
                col = dis_col(gen);
            } while (local_used_positions.find({row, col}) != local_used_positions.end());

            local_used_positions.insert({row, col});
            double value = dis_val(gen);
            local_elements.push_back(std::make_tuple(row, col, value));
        }

        #pragma omp critical
        {
            elements.insert(elements.end(), local_elements.begin(), local_elements.end());
        }
    }

    // Write to file
    for (const auto& elem : elements) {
        int row, col;
        double value;
        std::tie(row, col, value) = elem;
        file << (row + 1) << " " << (col + 1) << " " << std::fixed << std::setprecision(1) << value << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <output_file.mtx> <num_rows> <num_cols> <num_nonzeros>" << std::endl;
        return 1;
    }

    try {
        std::string filename = argv[1];
        int num_rows = std::stoi(argv[2]);
        int num_cols = std::stoi(argv[3]);
        int num_nonzeros = std::stoi(argv[4]);

        if (num_nonzeros > (num_rows * num_cols) / 2) {
            throw std::invalid_argument("Number of non-zero elements cannot exceed half number of matrix elements");
        }

        auto start = std::chrono::high_resolution_clock::now();

        generateRandomSparseMatrix(filename, num_rows, num_cols, num_nonzeros);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Matrix generated and saved to " << filename << std::endl;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}