#include <iostream>
#include <string>

bool run_vector_add(double* elapsed_microseconds);

int main(int argc, char** argv) {
    std::string mode;
    std::string protocol;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--mode" && index + 1 < argc) {
            mode = argv[++index];
        } else if (argument == "--protocol" && index + 1 < argc) {
            protocol = argv[++index];
        } else {
            return 2;
        }
    }
    if (mode != "correctness" && mode != "events") {
        return 2;
    }
    if (mode == "events" && protocol != "trusted-smoke-v1") {
        return 2;
    }
    double elapsed_microseconds = 0.0;
    if (!run_vector_add(&elapsed_microseconds)) {
        return 1;
    }
    if (mode == "events") {
        std::cout << "{\"value\":" << elapsed_microseconds
                  << ",\"unit\":\"us\",\"source\":\"cuda_events\","
                     "\"protocol_id\":\"trusted-smoke-v1\"}\n";
    } else {
        std::cout << "{\"correct\":true}\n";
    }
    return 0;
}
