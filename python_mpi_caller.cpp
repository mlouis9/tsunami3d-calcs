#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Check if we are running with exactly 12 processors
    if (world_size != 12) {
        std::cerr << "This application is meant to be run with 12 processors." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Prepare the command string
    std::stringstream cmd;
    cmd << "python create_and_run_cases.py " << world_rank;

    // Call the python script
    std::cout << cmd.str() << std::endl;
    
    int result = system(cmd.str().c_str());
    if (result != 0) {
        std::cerr << "Failed to execute the Python script for rank " << world_rank << std::endl;
    }

    MPI_Finalize();
    return 0;
}
