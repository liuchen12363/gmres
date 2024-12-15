#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono> // For measuring time
#include <unordered_map>

struct CSRMatrix {
    std::vector<double> values;
    std::vector<int> col_index;
    std::vector<int> row_ptr;
    int num_rows;
    int num_cols;
};

CSRMatrix readMatrixMarketToCSR(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    // Skip comments and header
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions and number of non-zero elements
    std::istringstream iss(line);
    int num_rows, num_cols, num_nonzeros;
    iss >> num_rows >> num_cols >> num_nonzeros;

    // Temporary storage for COO format
    std::vector<int> row_indices(num_nonzeros);
    std::vector<int> col_indices(num_nonzeros);
    std::vector<double> values(num_nonzeros);

    // Read the matrix data
    for (int i = 0; i < num_nonzeros; ++i) {
        file >> row_indices[i] >> col_indices[i] >> values[i];
        // Convert to zero-based index
        row_indices[i]--;
        col_indices[i]--;
    }

    // Initialize CSR structure
    CSRMatrix csr;
    csr.num_rows = num_rows;
    csr.num_cols = num_cols;
    csr.values = values;
    csr.col_index = col_indices;
    csr.row_ptr.resize(num_rows + 1, 0);

    // Count the number of entries in each row
    for (int i = 0; i < num_nonzeros; ++i) {
        csr.row_ptr[row_indices[i] + 1]++;
    }

    // Cumulative sum to get row_ptr
    for (int i = 0; i < num_rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    return csr;
}

CSRMatrix multiplyCSR(const CSRMatrix& A, const CSRMatrix& B) {
    if (A.num_cols != B.num_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    CSRMatrix C;
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.row_ptr.resize(C.num_rows + 1, 0);

    // Used to store non-zero elements and column indices for each row
    std::vector<std::unordered_map<int, double>> row_map(C.num_rows);

    // Iterate over each row of A
    for (int i = 0; i < A.num_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int a_col = A.col_index[j];
            double a_val = A.values[j];

            // Corresponding row in B for a_col
            for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; ++k) {
                int b_col = B.col_index[k];
                double b_val = B.values[k];

                // Accumulate results in the i-th row of the result matrix
                row_map[i][b_col] += a_val * b_val;
            }
        }
    }

    // Transfer results from row_map to CSR format
    for (int i = 0; i < C.num_rows; ++i) {
        for (const auto& [col, value] : row_map[i]) {
            if (value != 0.0) {
                C.values.push_back(value);
                C.col_index.push_back(col);
            }
        }
        C.row_ptr[i + 1] = C.values.size();
    }

    return C;
}

void writeCSRToMatrixMarket(const CSRMatrix& csr, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << csr.num_rows << " " << csr.num_cols << " " << csr.values.size() << "\n";

    for (int i = 0; i < csr.num_rows; ++i) {
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; ++j) {
            file << (i + 1) << " " << (csr.col_index[j] + 1) << " " << csr.values[j] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrixA.mtx> <matrixB.mtx> <result.mtx>" << std::endl;
        return 1;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        CSRMatrix A = readMatrixMarketToCSR(argv[1]);
        CSRMatrix B = readMatrixMarketToCSR(argv[2]);

        CSRMatrix C = multiplyCSR(A, B);

        writeCSRToMatrixMarket(C, argv[3]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}