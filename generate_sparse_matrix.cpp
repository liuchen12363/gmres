#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <iomanip>
#include <random> 
#include <chrono>

void generateRandomSparseMatrix(const std::string& filename, int num_rows, int num_cols, int num_nonzeros) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write Matrix Market header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << num_rows << " " << num_cols << " " << num_nonzeros << "\n";

    std::set<std::pair<int, int>> used_positions;
    std::random_device rd;
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_int_distribution<> dis_row(0, num_rows - 1);
    std::uniform_int_distribution<> dis_col(0, num_cols - 1);
    std::uniform_real_distribution<> dis_val(0.0, 100.0);

    for (int i = 0; i < num_nonzeros; ++i) {
        int row, col;
        do {
            row = dis_row(gen);
            col = dis_col(gen);
        } while (used_positions.find({row, col}) != used_positions.end());

        used_positions.insert({row, col});
        double value = dis_val(gen); // Random value between 0 and 100

        // Write to file (1-based index) with one decimal place
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

        if (num_nonzeros > (num_rows * num_cols)/2) {
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