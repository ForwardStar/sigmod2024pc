CXX=g++
CXXFLAGS=-I. -std=c++11 -O3
TARGET=test
SRC=baseline.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)