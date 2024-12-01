# Compiler and flags
CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -O2 `pkg-config --cflags opencv4`
LDFLAGS = -pthread `pkg-config --libs opencv4` -lmpi

# Target executable
TARGET = image_processor

# Source and object files
SRCS = image_processor.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files and outputs
clean:
	rm -f $(OBJS) $(TARGET) output_sequence.png output_pthread.png output_mpi.png

# Run program in different modes
run-sequence:
	./$(TARGET) input.png 1

run-pthread:
	./$(TARGET) input.png 2

run-mpi:
	mpirun -np 4 ./$(TARGET) input.png 3
