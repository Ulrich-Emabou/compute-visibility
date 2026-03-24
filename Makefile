# Makefile pour GitHub Actions
CXX = g++
CXXFLAGS = -std=c++14 -O3 -D_GLIBCXX_USE_CXX11_ABI=1 -DARMA_USE_HDF5
INCLUDES = -I./include
LIBS = -L./lib -lembree -ltbb -larmadillo -loctomath -loctomap -lhdf5 -ljpeg -lpng -ltiff

default:
	$(CXX) compute_visibility.cc -o compute_visibility.exe $(CXXFLAGS) $(INCLUDES) $(LIBS)