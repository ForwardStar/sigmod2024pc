CXX=g++
CXXFLAGS=-I. -std=c++11 -O3
TARGET=test
SRC=main.cpp
COMPARATORTARGET=comparator
COMPARATORSRC=comparator.cpp
BASELINETARGET=baseline
BASELINESOURCE=baseline.cpp

all: $(TARGET) $(COMPARATORTARGET) $(BASELINETARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(COMPARATORTARGET): $(COMPARATORSRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BASELINETARGET): $(BASELINESOURCE)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)