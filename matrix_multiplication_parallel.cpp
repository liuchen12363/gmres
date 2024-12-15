#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono> // For measuring time
#include <omp.h> 
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

    std::vector<std::vector<double>> temp_values(C.num_rows);
    std::vector<std::vector<int>> temp_col_indices(C.num_rows);

    // Use OpenMP to parallelize the outer loop
    #pragma omp parallel
    {
        std::vector<double> row_values(B.num_cols, 0.0);
        std::vector<int> local_col_indices;
        std::vector<double> local_values;

        // Each thread will have its own private copy of row_counts
        std::vector<int> row_counts(C.num_rows + 1, 0);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.num_rows; ++i) {
            std::fill(row_values.begin(), row_values.end(), 0.0);
            local_col_indices.clear();
            local_values.clear();

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int a_col = A.col_index[j];
                double a_val = A.values[j];
                for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; ++k) {
                    int b_col = B.col_index[k];
                    double b_val = B.values[k];
                    row_values[b_col] += a_val * b_val;
                }
            }

            for (int col = 0; col < B.num_cols; ++col) {
                if (row_values[col] != 0.0) {
                    local_col_indices.push_back(col);
                    local_values.push_back(row_values[col]);
                }
            }

            // Store results in temporary vectors
            temp_values[i] = std::move(local_values);
            temp_col_indices[i] = std::move(local_col_indices);

            // Accumulate the count of non-zero elements for this row
            row_counts[i + 1] = temp_values[i].size();
        }

        // Combine the row counts from all threads
        #pragma omp critical
        {
            for (int i = 0; i < C.num_rows; ++i) {
                C.row_ptr[i + 1] += row_counts[i + 1];
            }
        }
    }

    // Compute the cumulative sum to get row_ptr
    for (int i = 0; i < C.num_rows; ++i) {
        C.row_ptr[i + 1] += C.row_ptr[i];
    }

    // Merge all threads' results
    for (int i = 0; i < C.num_rows; ++i) {
        C.values.insert(C.values.end(), temp_values[i].begin(), temp_values[i].end());
        C.col_index.insert(C.col_index.end(), temp_col_indices[i].begin(), temp_col_indices[i].end());
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